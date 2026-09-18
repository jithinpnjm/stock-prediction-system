from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import optuna
import polars as pl
import torch
import yaml
from sklearn.metrics import log_loss
from sklearn.preprocessing import RobustScaler

from src.mlops.lineage import Lineage
from src.mlops.mlflow_utils import log_lineage
from src.models.sequence_data import build_sequences
from src.models.sequence_training import sequence_predict_proba, train_sequence_classifier
from src.models.tcn import build_tcn
from src.models.transformer import build_transformer
from src.research.experiment_budget import ExperimentBudget
from src.validation.purged_cv import PurgedTimeSeriesSplit

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "seq_train", Path(__file__).parent / "07_sequence_train.py"
)
_seq_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_seq_train)
SEQUENCE_FEATURE_COLUMNS = _seq_train.SEQUENCE_FEATURE_COLUMNS


def _walk_forward_log_loss(
    model_family: str,
    params: dict,
    batch,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_features: int,
    min_train_rows: int,
    epochs: int,
    patience: int,
    clip_z: float,
) -> tuple[float, int]:
    losses = []
    n_folds_used = 0
    for train_idx, val_idx in folds:
        if len(train_idx) < min_train_rows:
            continue
        n_folds_used += 1
        scaler = RobustScaler().fit(batch.X[train_idx].reshape(-1, n_features))
        scaled_X = (
            np.clip(scaler.transform(batch.X.reshape(-1, n_features)), -clip_z, clip_z)
            .reshape(batch.X.shape)
            .astype(np.float32)
        )
        if model_family == "tcn":
            model = build_tcn(
                n_features,
                channels=params["channels"],
                kernel_size=params["kernel_size"],
                dropout=params["dropout"],
            )
            X = scaled_X.transpose(0, 2, 1)
        else:
            model = build_transformer(
                n_features,
                d_model=params["d_model"],
                n_heads=params["n_heads"],
                n_layers=params["n_layers"],
                dropout=params["dropout"],
            )
            X = scaled_X
        result = train_sequence_classifier(
            model,
            X[train_idx],
            batch.y[train_idx],
            X[val_idx],
            batch.y[val_idx],
            epochs=epochs,
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            patience=patience,
        )
        probs = sequence_predict_proba(result.model, X[val_idx])
        losses.append(log_loss(batch.y[val_idx], probs, labels=[-1, 0, 1]))
    if not losses:
        raise RuntimeError("no valid folds for this trial")
    return float(np.mean(losses)), n_folds_used


def run(model_family: str, n_trials: int, epochs: int):
    cfg = yaml.safe_load(Path("configs/models/sequence.yaml").read_text())
    dataset = pl.read_parquet("data/gold/dataset_v1.parquet").sort("timestamp")
    names = [c for c in SEQUENCE_FEATURE_COLUMNS if c in dataset.columns]
    usable = dataset.drop_nulls(subset=names + ["label", "event_end_timestamp"]).sort("timestamp")

    sequence_length = int(cfg["sequence_length"])
    batch = build_sequences(
        usable.select(names).to_numpy(),
        usable["label"].to_numpy(),
        usable["timestamp"].dt.epoch(time_unit="ns").to_numpy(),
        usable["event_end_timestamp"].dt.epoch(time_unit="ns").to_numpy(),
        sequence_length=sequence_length,
    )
    embargo_ns = int(cfg.get("embargo_bars", 75)) * 5 * 60 * 1_000_000_000
    split = PurgedTimeSeriesSplit(int(cfg.get("n_splits", 4)), embargo=embargo_ns)
    folds = list(split.split(batch.timestamps, batch.event_end))
    n_features = batch.X.shape[2]
    min_train_rows = int(cfg.get("min_train_rows", 500))
    patience = int(cfg.get("patience", 8))
    clip_z = float(cfg.get("clip_z", 8.0))

    budget = ExperimentBudget(max_trials=n_trials, experiment_family=f"sequence_{model_family}")
    mlflow.set_experiment("BankNifty_Sequence_Sweeps")
    device_used = "cuda" if torch.cuda.is_available() else "cpu"

    def objective(trial: optuna.Trial) -> float:
        budget.validate_trial(trial.number)
        if model_family == "tcn":
            depth = trial.suggest_int("depth", 2, 5)
            width = trial.suggest_categorical("width", [32, 64, 96, 128])
            params = {
                "channels": tuple(width for _ in range(depth)),
                "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.4),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            }
        else:
            n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
            d_model = n_heads * trial.suggest_categorical("d_model_per_head", [16, 24, 32])
            params = {
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": trial.suggest_int("n_layers", 2, 6),
                "dropout": trial.suggest_float("dropout", 0.0, 0.4),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            }
        with mlflow.start_run(run_name=f"{model_family}_trial_{trial.number}", nested=True):
            mlflow.log_params({**params, "channels": str(params.get("channels", ""))})
            mlflow.log_param("device", device_used)
            log_lineage(
                Lineage.create(
                    dataset_id="banknifty_5m_v1",
                    feature_version="sequence_early_session_v1",
                    label_version="triple_barrier_200_70_v1",
                    validation_version="purged_time_series_v1",
                ).to_dict()
            )
            loss, n_folds_used = _walk_forward_log_loss(
                model_family,
                params,
                batch,
                folds,
                n_features,
                min_train_rows,
                epochs,
                patience,
                clip_z,
            )
            mlflow.log_metric("cv_log_loss", loss)
            mlflow.log_metric("n_folds_used", n_folds_used)
            mlflow.log_param("trial_number", trial.number)
        return loss

    with mlflow.start_run(run_name=f"{model_family}_sweep_parent"):
        mlflow.log_params({"model_family": model_family, "n_trials": n_trials, "epochs": epochs})
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        mlflow.log_metric("best_cv_log_loss", float(study.best_value))
        for k, v in study.best_params.items():
            mlflow.log_param(f"best_{k}", v)
        print(f"{model_family} sweep best_value={study.best_value:.5f}")
        print(f"{model_family} sweep best_params={study.best_params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tcn", "transformer"], default="tcn")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    run(args.model, args.trials, args.epochs)
