from __future__ import annotations

import polars as pl


def add_session_context_features(df: pl.DataFrame) -> pl.DataFrame:
    out = df.sort("timestamp").with_columns(
        pl.col("timestamp").dt.date().alias("_date")
    )
    daily = (
        out.group_by("_date")
        .agg(
            pl.col("open").first().alias("_day_open"),
            pl.col("high").max().alias("_day_high"),
            pl.col("low").min().alias("_day_low"),
            pl.col("close").last().alias("_day_close"),
            pl.col("volume").sum().alias("_day_volume"),
        )
        .sort("_date")
        .with_columns(
            pl.col("_day_open").shift(1).alias("_prev_day_open"),
            pl.col("_day_high").shift(1).alias("_prev_day_high"),
            pl.col("_day_low").shift(1).alias("_prev_day_low"),
            pl.col("_day_close").shift(1).alias("_prev_day_close"),
            pl.col("_day_volume").shift(1).alias("_prev_day_volume"),
        )
        .with_columns(
            (
                (pl.col("_prev_day_close") - pl.col("_prev_day_open"))
                / (pl.col("_prev_day_open") + 1e-9)
            ).alias("_prev_day_return"),
            (pl.col("_prev_day_high") - pl.col("_prev_day_low")).alias(
                "_prev_day_range"
            ),
        )
    )
    return (
        out.join(daily, on="_date", how="left")
        .with_columns(
            (pl.col("open") - pl.col("_prev_day_close")).alias(
                "f_gap_from_prev_close"
            ),
            (
                (pl.col("open") - pl.col("_prev_day_close"))
                / (pl.col("_prev_day_close") + 1e-9)
            ).alias("f_gap_from_prev_close_pct"),
            (pl.col("close") - pl.col("_prev_day_high")).alias(
                "f_distance_prev_high"
            ),
            (pl.col("close") - pl.col("_prev_day_low")).alias(
                "f_distance_prev_low"
            ),
            (
                (pl.col("close") - pl.col("_prev_day_low"))
                / (pl.col("_prev_day_high") - pl.col("_prev_day_low") + 1e-9)
            ).alias("f_prev_day_range_position"),
            pl.col("_prev_day_return").alias("f_prev_day_return"),
            pl.col("_prev_day_range").alias("f_prev_day_range"),
            pl.col("high").cum_max().over("_date").alias("f_session_high_so_far"),
            pl.col("low").cum_min().over("_date").alias("f_session_low_so_far"),
            (
                pl.col("high").cum_max().over("_date") - pl.col("close")
            ).alias("f_intraday_high_pullback"),
            (
                pl.col("close") - pl.col("low").cum_min().over("_date")
            ).alias("f_intraday_low_rebound"),
        )
        .drop(
            [
                "_date",
                "_day_open",
                "_day_high",
                "_day_low",
                "_day_close",
                "_day_volume",
                "_prev_day_open",
                "_prev_day_high",
                "_prev_day_low",
                "_prev_day_close",
                "_prev_day_volume",
                "_prev_day_return",
                "_prev_day_range",
            ]
        )
    )
