from __future__ import annotations

import polars as pl


def validate_prediction_alignment(
    predictions: pl.DataFrame,
    source: pl.DataFrame,
) -> None:
    if predictions.is_empty():
        raise ValueError("Predictions are empty")
    if predictions.get_column("timestamp").n_unique() != predictions.height:
        raise ValueError("Predictions contain duplicate timestamps")
    source_ts = set(source.get_column("timestamp").to_list())
    missing = [ts for ts in predictions.get_column("timestamp").to_list() if ts not in source_ts]
    if missing:
        raise ValueError(f"Prediction timestamps not found in source: {missing[:3]}")
