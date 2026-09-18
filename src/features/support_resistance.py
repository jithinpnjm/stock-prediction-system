from __future__ import annotations

import polars as pl


def add_support_resistance_features(
    df: pl.DataFrame,
    lookback: int = 48,
    touch_tolerance_points: float = 10.0,
) -> pl.DataFrame:
    out = df.sort("timestamp").with_columns(pl.col("timestamp").dt.date().alias("_session_date"))
    resistance = pl.col("high").shift(1).rolling_max(lookback).over("_session_date")
    support = pl.col("low").shift(1).rolling_min(lookback).over("_session_date")
    r_touch = (
        ((pl.col("high") - resistance).abs() <= touch_tolerance_points)
        .cast(pl.Int8)
        .rolling_sum(lookback, min_samples=1)
        .over("_session_date")
    )
    s_touch = (
        ((pl.col("low") - support).abs() <= touch_tolerance_points)
        .cast(pl.Int8)
        .rolling_sum(lookback, min_samples=1)
        .over("_session_date")
    )
    return out.with_columns(
        resistance.alias("f_resistance"),
        support.alias("f_support"),
        (pl.col("close") - resistance).alias("f_distance_to_resistance"),
        (pl.col("close") - support).alias("f_distance_to_support"),
        r_touch.alias("f_resistance_touches"),
        s_touch.alias("f_support_touches"),
        ((pl.col("close") - support) / (resistance - support + 1e-9)).alias(
            "f_position_in_sr_range"
        ),
    ).drop("_session_date")
