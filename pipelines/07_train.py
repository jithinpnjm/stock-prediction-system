from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import log_loss

from src.common.config import load_yaml
from src.mlops.lineage import build_lineage
from src.mlops.mlflow_utils import log_lineage, log_metrics
from src.mlops.reproducibility import seed_everything
from src.models.baselines import make_lightgbm_classifier
from src.validation.purged_cv import PurgedWalkForwardSplit

CLASS_TO_INDEX = {-1: 0, 0: 1, 1: 2}


def _encode(y: np.ndarray) -> np.ndarray:
    return np.asarray([CLASS_TO_INDEX[int(v)] for v in y], dtype=int)


def run() -> None:
    cfg = load_yaml(os.getenv("MODEL_CONFIG", "configs/models/lgbm_baseline.yaml"))
    val_cfg = load_yaml(
        os.getenv("VALIDATION_CONFIG", "configs/validation/default.yaml")
    )
    exp_cfg = load_yaml(
        os.getenv("EXPERIMENT_CONFIG", "configs/experiments/default.yaml")
    )
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    frame = pl.read_parquet("data/ml/training_frame.parquet").sort("timestamp")
    feature_columns = json.loads(
        Path("data/ml/feature_columns.json").read_text(encoding="utf-8")
    )
    if frame.get_column("timestamp").n_unique() != frame.height:
        raise ValueError("Training frame contains duplicate timestamps")

    holdout_fraction = float(cfg.get("holdout_fraction", 0.20))
    split_row = int(frame.height * (1.0 - holdout_fraction))
    if split_row <= 100 or split_row >= frame.height:
        raise ValueError("Development/holdout sizes are invalid")

    holdout_start = frame["timestamp"][split_row]
    # Prevent development labels from reaching into the frozen holdout period.
    dev = frame.filter(pl.col("event_end") < holdout_start)
    holdout = frame.filter(pl.col("timestamp") >= holdout_start)
    if dev.height <= 100 or holdout.is_empty():
        raise ValueError("Holdout purge left insufficient development data")

    X_dev = dev.select(feature_columns).to_numpy()
    y_dev = _encode(dev.get_column("label").to_numpy())
    starts = dev.get_column("timestamp").to_list()
    ends = dev.get_column("event_end").to_list()

    mlflow.set_experiment(exp_cfg["experiment_name"])
    with mlflow.start_run(run_name="lgbm_walk_forward") as run:
        log_lineage(
            build_lineage(
                feature_version=exp_cfg["feature_version"],
                label_version=exp_cfg["label_version"],
                dvc_revision=exp_cfg["dataset_version"],
            )
        )
        mlflow.log_param("holdout_fraction", holdout_fraction)
        mlflow.log_param("feature_count", len(feature_columns))
        mlflow.log_param("dev_rows", dev.height)
        mlflow.log_param("holdout_rows", holdout.height)

        splitter = PurgedWalkForwardSplit(
            n_splits=int(val_cfg["n_splits"]),
            embargo=pd.Timedelta(minutes=int(val_cfg["embargo_minutes"])),
        )
        oof_parts = []
        fold_metrics = []

        params = {
            "n_estimators": int(cfg.get("n_estimators", 1000)),
            "learning_rate": float(cfg.get("learning_rate", 0.03)),
            "num_leaves": int(cfg.get("num_leaves", 31)),
            "max_depth": int(cfg.get("max_depth", -1)),
            "subsample": float(cfg.get("subsample", 0.8)),
            "colsample_bytree": float(cfg.get("colsample_bytree", 0.8)),
            "reg_alpha": float(cfg.get("reg_alpha", 0.1)),
            "reg_lambda": float(cfg.get("reg_lambda", 0.5)),
            "class_weight": cfg.get("class_weight", "balanced"),
        }

        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(starts, ends), start=1
        ):
            model = make_lightgbm_classifier(seed=seed + fold, params=params)
            model.fit(
                X_dev[train_idx],
                y_dev[train_idx],
                eval_set=[(X_dev[val_idx], y_dev[val_idx])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            proba = model.predict_proba(X_dev[val_idx])
            fold_metrics.append(
                float(log_loss(y_dev[val_idx], proba, labels=[0, 1, 2]))
            )
            oof_parts.append(
                pl.DataFrame(
                    {
                        "timestamp": [starts[i] for i in val_idx],
                        "y_true": [int(y_dev[i]) for i in val_idx],
                        "p_short": proba[:, 0],
                        "p_flat": proba[:, 1],
                        "p_long": proba[:, 2],
                        "fold": fold,
                    }
                )
            )

        oof = pl.concat(oof_parts).unique("timestamp").sort("timestamp")
        Path("data/ml").mkdir(parents=True, exist_ok=True)
        oof.write_parquet("data/ml/oof_predictions.parquet", compression="zstd")
        log_metrics(
            {f"fold_{i + 1}_logloss": value for i, value in enumerate(fold_metrics)}
        )
        log_metrics({"mean_oof_logloss": float(np.mean(fold_metrics))})

        final_model = make_lightgbm_classifier(seed=seed, params=params)
        final_model.fit(X_dev, y_dev)
        Path("artifacts/models").mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": final_model,
            "feature_columns": feature_columns,
            "class_order": [-1, 0, 1],
            "dev_rows": dev.height,
            "holdout_start": holdout_start,
            "config": cfg,
        }
        joblib.dump(artifact, "artifacts/models/lgbm_dev.joblib")
        mlflow.lightgbm.log_model(final_model, artifact_path="model")
        mlflow.log_artifact("data/ml/oof_predictions.parquet")
        mlflow.log_artifact("artifacts/models/lgbm_dev.joblib")
        mlflow.log_param("run_id", run.info.run_id)


if __name__ == "__main__":
    run()
