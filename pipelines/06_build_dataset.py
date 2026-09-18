from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl


METADATA = {
    "session_date",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "label",
    "event_end",
    "entry_time",
    "barrier_type",
    "barrier_time",
    "path_complete",
    "long_outcome",
    "short_outcome",
    "time_to_barrier_min",
    "mfe_long",
    "mae_long",
    "mfe_short",
    "mae_short",
    "mfe_time_long",
    "mae_time_long",
    "mfe_time_short",
    "mae_time_short",
}


def run() -> None:
    df = pl.read_parquet("data/gold/dataset_v1.parquet")
    if "path_complete" in df.columns:
        df = df.filter(pl.col("path_complete"))
    if "barrier_type" in df.columns:
        df = df.filter(pl.col("barrier_type") != "ambiguous")

    candidates = []
    for column, dtype in df.schema.items():
        if column in METADATA:
            continue
        if dtype.is_numeric():
            candidates.append(column)

    if not candidates:
        raise ValueError("No numeric features available")

    # Missing values are never silently interpreted as zero. We remove only rows
    # that lack a value for a feature used by the model.
    train = df.drop_nulls(subset=candidates)
    train = train.select(["timestamp", "event_end", "label", *candidates])

    os.makedirs("data/ml", exist_ok=True)
    train.write_parquet("data/ml/training_frame.parquet", compression="zstd")
    train.select(candidates).write_parquet("data/ml/X.parquet", compression="zstd")
    train.select("label").write_parquet("data/ml/y.parquet", compression="zstd")
    Path("data/ml/feature_columns.json").write_text(
        json.dumps(candidates, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Training frame: {train.height} rows x {len(candidates)} features")


if __name__ == "__main__":
    run()
