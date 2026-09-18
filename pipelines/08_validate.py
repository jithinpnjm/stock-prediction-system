from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import balanced_accuracy_score, classification_report, log_loss

from src.common.config import load_yaml
from src.validation.calibration import (
    MulticlassProbabilityCalibrator,
    expected_calibration_error,
)


def encode_labels(values: np.ndarray) -> np.ndarray:
    mapping = {-1: 0, 0: 1, 1: 2}
    return np.asarray([mapping[int(value)] for value in values], dtype=int)


def run() -> None:
    val_cfg = load_yaml(
        os.getenv("VALIDATION_CONFIG", "configs/validation/default.yaml")
    )
    oof = pl.read_parquet("data/ml/oof_predictions.parquet").sort("timestamp")
    probabilities = oof.select(
        ["p_short", "p_flat", "p_long"]
    ).to_numpy()
    y = oof["y_true"].to_numpy()

    raw_pred = probabilities.argmax(axis=1)
    raw_metrics = {
        "logloss": float(
            log_loss(y, probabilities, labels=[0, 1, 2])
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, raw_pred)
        ),
        "ece": float(
            expected_calibration_error(probabilities, y)
        ),
    }

    split = int(len(oof) * 0.5)
    if split < 20 or len(oof) - split < 20:
        raise SystemExit(
            "Not enough OOF rows for temporally separated calibration"
        )

    calibrator = MulticlassProbabilityCalibrator()
    calibrator.fit(probabilities[:split], y[:split])
    calibrated_eval = calibrator.predict_proba(probabilities[split:])
    calibrated_y = y[split:]
    calibrated_pred = calibrated_eval.argmax(axis=1)

    calibrated_oof = oof.slice(split).with_columns(
        [
            pl.Series("p_short", calibrated_eval[:, 0]),
            pl.Series("p_flat", calibrated_eval[:, 1]),
            pl.Series("p_long", calibrated_eval[:, 2]),
        ]
    )

    Path("data/ml").mkdir(parents=True, exist_ok=True)
    calibrated_oof.write_parquet(
        "data/ml/calibrated_oof_predictions.parquet",
        compression="zstd",
    )

    Path("artifacts/models").mkdir(parents=True, exist_ok=True)
    joblib.dump(
        calibrator,
        "artifacts/models/oof_calibrator.joblib",
    )

    calibrated_metrics = {
        "evaluation_rows": int(len(calibrated_y)),
        "logloss": float(
            log_loss(calibrated_y, calibrated_eval, labels=[0, 1, 2])
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                calibrated_y, calibrated_pred
            )
        ),
        "ece": float(
            expected_calibration_error(
                calibrated_eval, calibrated_y
            )
        ),
    }

    report = {
        "status": "structural_pass",
        "raw_oof": raw_metrics,
        "calibrated_oof": calibrated_metrics,
        "calibration_training_rows": split,
        "holdout_evaluated": False,
        "holdout": None,
        "classification_report_calibrated": classification_report(
            calibrated_y,
            calibrated_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    allow_holdout = bool(
        val_cfg.get("allow_holdout_eval", False)
    ) or os.getenv("ALLOW_HOLDOUT_EVAL") == "1"

    if allow_holdout:
        artifact = joblib.load(
            "artifacts/models/lgbm_dev.joblib"
        )
        frame = pl.read_parquet(
            "data/ml/training_frame.parquet"
        ).sort("timestamp")
        holdout_start = artifact["holdout_start"]
        holdout = frame.filter(
            pl.col("timestamp") >= holdout_start
        )
        features = artifact["feature_columns"]
        model = artifact["model"]
        holdout_probabilities = model.predict_proba(
            holdout.select(features).to_numpy()
        )
        holdout_y = encode_labels(
            holdout["label"].to_numpy()
        )
        calibrated_holdout = calibrator.predict_proba(
            holdout_probabilities
        )
        holdout_pred = calibrated_holdout.argmax(axis=1)

        report["holdout_evaluated"] = True
        report["holdout"] = {
            "rows": int(holdout.height),
            "logloss": float(
                log_loss(
                    holdout_y,
                    calibrated_holdout,
                    labels=[0, 1, 2],
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    holdout_y,
                    holdout_pred,
                )
            ),
            "ece": float(
                expected_calibration_error(
                    calibrated_holdout,
                    holdout_y,
                )
            ),
        }

    Path("artifacts/validation").mkdir(
        parents=True, exist_ok=True
    )
    Path(
        "artifacts/validation/report.json"
    ).write_text(
        json.dumps(
            report, indent=2, default=str
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
