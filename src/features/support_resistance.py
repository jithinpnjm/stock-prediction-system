from __future__ import annotations

import polars as pl


def add_support_resistance_features(
    df: pl.DataFrame,
    lookback: int = 48,
    touch_tolerance_points: float = 10.0,
) -> pl.DataFrame:
    out = df.sort("timestamp")
    resistance = pl.col("high").shift(1).rolling_max(lookback)
    support = pl.col("low").shift(1).rolling_min(lookback)
    return out.with_columns(
        resistance.alias("f_resistance"),
        support.alias("f_support"),
        (pl.col("close") - resistance).alias("f_distance_to_resistance"),
        (pl.col("close") - support).alias("f_distance_to_support"),
        (
            (pl.col("high") - resistance).abs() <= touch_tolerance_points
        ).cast(pl.Int8).rolling_sum(lookback).alias("f_resistance_touches"),
        (
            (pl.col("low") - support).abs() <= touch_tolerance_points
        ).cast(pl.Int8).rolling_sum(lookback).alias("f_support_touches"),
    )
