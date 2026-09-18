from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import StandardScaler

from src.common.config import load_yaml
from src.mlops.reproducibility import seed_everything
from src.models.sequences import (
    build_sequences,
    chronological_sequence_split,
)
from src.models.sequence_training import (
    train_sequence_classifier,
)
from src.models.torch_models import (
    TCNClassifier,
    TransformerClassifier,
)


def encode_labels(
    values: np.ndarray,
) -> np.ndarray:
    mapping = {-1: 0, 0: 1, 1: 2}
    return np.asarray(
        [mapping[int(value)] for value in values],
        dtype=np.int64,
    )


def main() -> None:
    model_name = os.getenv(
        "SEQUENCE_MODEL",
        "tcn",
    ).lower()

    config = load_yaml(
        os.getenv(
            "MODEL_CONFIG",
            "configs/models/lgbm_baseline.yaml",
        )
    )

    seed = int(
        config.get("seed", 42)
    )
    seed_everything(seed)

    frame = (
        pl.read_parquet(
            "data/ml/training_frame.parquet"
        )
        .sort("timestamp")
    )

    features = json.loads(
        Path(
            "data/ml/feature_columns.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    holdout_fraction = float(
        config.get(
            "holdout_fraction",
            0.20,
        )
    )
    dev_rows = int(
        frame.height
        * (1.0 - holdout_fraction)
    )
    if dev_rows <= 100:
        raise SystemExit(
            "Not enough development rows for "
            "sequence research"
        )

    dev = frame.head(dev_rows)

    X = (
        dev.select(features)
        .to_numpy()
        .astype(np.float32)
    )
    y = encode_labels(
        dev["label"].to_numpy()
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(
        np.float32
    )

    sequences, sequence_labels, sequence_ts = (
        build_sequences(
            X,
            y,
            dev["timestamp"].to_list(),
            window=32,
        )
    )

    if len(sequences) < 100:
        raise SystemExit(
            "Not enough contiguous development "
            "sequences"
        )

    X_train, y_train, X_val, y_val = (
        chronological_sequence_split(
            sequences,
            sequence_labels,
            validation_fraction=0.20,
        )
    )

    if model_name == "tcn":
        model = TCNClassifier(
            input_dim=sequences.shape[2]
        )
    elif model_name == "transformer":
        model = TransformerClassifier(
            input_dim=sequences.shape[2]
        )
    else:
        raise ValueError(
            "SEQUENCE_MODEL must be tcn or transformer"
        )

    trained, history = (
        train_sequence_classifier(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            seed=seed,
            device=(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            ),
        )
    )

    model_dir = Path(
        "artifacts/models"
    )
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state": trained.state_dict(),
            "model_name": model_name,
            "input_dim": sequences.shape[2],
            "feature_columns": features,
            "window": 32,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "last_sequence_timestamp": sequence_ts[-1],
            "history": history,
            "dev_rows": dev_rows,
        },
        model_dir
        / f"{model_name}_dev.pt",
    )

    joblib.dump(
        scaler,
        model_dir
        / f"{model_name}_scaler.joblib",
    )

    print(
        f"{model_name} sequence training complete: "
        f"{len(X_train)} train / "
        f"{len(X_val)} validation sequences"
    )


if __name__ == "__main__":
    main()
