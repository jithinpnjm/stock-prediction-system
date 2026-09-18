from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import balanced_accuracy_score, classification_report, log_loss

from src.common.config import load_yaml
from src.validation.calibration import expected_calibration_error


def run() -> None:
    val_cfg = load_yaml(os.getenv("VALIDATION_CONFIG", "configs/validation/default.yaml"))
    oof = pl.read_parquet("data/ml/oof_predictions.parquet")
    proba = oof.select(["p_short", "p_flat", "p_long"]).to_numpy()
    y = oof["y_true"].to_numpy()

    y_pred = proba.argmax(axis=1)
    metrics = {
        "oof_logloss": float(log_loss(y, proba, labels=[0, 1, 2])),
        "oof_balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "oof_ece": float(expected_calibration_error(proba, y)),
        "oof_rows": int(len(y)),
    }

    report = {
        "status": "structural_pass",
        "oof": metrics,
        "holdout_evaluated": False,
        "holdout": None,
        "classification_report": classification_report(
            y, y_pred, output_dict=True, zero_division=0
        ),
    }

    if bool(val_cfg.get("allow_holdout_eval", False)) or os.getenv("ALLOW_HOLDOUT_EVAL") == "1":
        artifact = joblib.load("artifacts/models/lgbm_dev.joblib")
        model = artifact["model"]
        features = artifact["feature_columns"]
        frame = pl.read_parquet("data/ml/training_frame.parquet")
        dev_rows = int(artifact["dev_rows"])
        holdout = frame.slice(dev_rows)
        X_holdout = holdout.select(features).to_numpy()
        y_holdout = np.asarray([{-1: 0, 0: 1, 1: 2}[int(v)] for v in holdout["label"].to_numpy()])
        p_holdout = model.predict_proba(X_holdout)
        y_holdout_pred = p_holdout.argmax(axis=1)
        report["holdout_evaluated"] = True
        report["holdout"] = {
            "rows": len(y_holdout),
            "logloss": float(log_loss(y_holdout, p_holdout, labels=[0, 1, 2])),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_holdout, y_holdout_pred)
            ),
            "ece": float(expected_calibration_error(p_holdout, y_holdout)),
        }

    Path("artifacts/validation").mkdir(parents=True, exist_ok=True)
    Path("artifacts/validation/report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
