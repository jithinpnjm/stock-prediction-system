from __future__ import annotations

import json
from pathlib import Path

import polars as pl

OUTCOME_COLUMNS = {
    "label",
    "event_end_timestamp",
    "barrier_timestamp",
    "target_points",
    "stop_points",
    "max_horizon_minutes",
    "mfe_long_points",
    "mae_long_points",
    "mfe_short_points",
    "mae_short_points",
    "mfe_points",
    "mae_points",
    "time_to_barrier_seconds",
}


def run():
    df = pl.read_parquet("data/gold/dataset_v1.parquet")
    feature_columns = [c for c in df.columns if c.startswith("f_") and c not in OUTCOME_COLUMNS]
    if not feature_columns:
        raise ValueError("No feature columns found")
    usable = df.drop_nulls(subset=feature_columns + ["label", "event_end_timestamp"]).sort(
        "timestamp"
    )
    if usable.is_empty():
        raise ValueError("No rows remain after enforcing feature availability")
    Path("data/ml").mkdir(parents=True, exist_ok=True)
    usable.select(["timestamp", "event_end_timestamp", "label", *feature_columns]).write_parquet(
        "data/ml/training_dataset.parquet"
    )
    Path("data/ml/feature_schema.json").write_text(
        json.dumps(
            {"feature_columns": feature_columns, "feature_count": len(feature_columns)}, indent=2
        )
        + "\n"
    )
    print(f"trainable rows={usable.height}, features={len(feature_columns)}")


if __name__ == "__main__":
    run()
