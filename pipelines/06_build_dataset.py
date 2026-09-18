from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl


OUTCOME_COLUMNS = {
    "label",
    "event_end_timestamp",
    "barrier_timestamp",
    "barrier_type",
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
    feature_columns = [
        column
        for column in df.columns
        if column.startswith("f_") and column not in OUTCOME_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("No feature columns found")

    required = feature_columns + ["timestamp", "label", "event_end_timestamp"]
    usable = df.drop_nulls(subset=required).sort("timestamp")

    feature_matrix = usable.select(feature_columns).to_numpy()
    finite_mask = np.isfinite(feature_matrix).all(axis=1)
    usable = usable.filter(pl.Series("finite_features", finite_mask))

    labels = usable["label"].to_numpy()
    if not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("labels must be one of -1, 0, 1")

    if usable.filter(
        pl.col("event_end_timestamp") <= pl.col("timestamp")
    ).height:
        raise ValueError("event_end_timestamp must be after event timestamp")

    if usable.is_empty():
        raise ValueError("No rows remain after enforcing feature availability")

    usable.select(
        ["timestamp", "event_end_timestamp", "label", *feature_columns]
    ).write_parquet("data/ml/training_dataset.parquet")

    Path("data/ml").mkdir(parents=True, exist_ok=True)
    Path("data/ml/feature_schema.json").write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "feature_count": len(feature_columns),
                "feature_version": "price_action_v4",
            },
            indent=2,
        )
        + "
"
    )
    print(
        f"trainable rows={usable.height}, "
        f"features={len(feature_columns)}"
    )


if __name__ == "__main__":
    run()
