from __future__ import annotations

import polars as pl


def add_candle_geometry_features(df: pl.DataFrame) -> pl.DataFrame:
    eps=1e-9
    return (
        df.sort("timestamp")
        .with_columns(
            (pl.col("high") - pl.col("low")).alias("f_range"),
            (pl.col("close") - pl.col("open")).alias("f_body"),
            (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("f_return_1"),
            pl.max_horizontal("open", "close").alias("_top"),
            pl.min_horizontal("open", "close").alias("_bottom"),
        )
        .with_columns(
            (pl.col("high") - pl.col("_top")).alias("f_upper_wick"),
            (pl.col("_bottom") - pl.col("low")).alias("f_lower_wick"),
            (
                (pl.col("close") - pl.col("low"))
                / (pl.col("f_range") + eps)
            ).alias("f_close_location"),
        )
        .with_columns(
            (pl.col("f_body").abs() / (pl.col("f_range") + eps)).alias("f_body_pct"),
            (pl.col("f_upper_wick") / (pl.col("f_range") + eps)).alias("f_upper_wick_pct"),
            (pl.col("f_lower_wick") / (pl.col("f_range") + eps)).alias("f_lower_wick_pct"),
            pl.when(pl.col("close") > pl.col("open")).then(1)
            .when(pl.col("close") < pl.col("open")).then(-1)
            .otherwise(0).alias("f_direction"),
            (pl.col("high") - pl.col("low")) / (pl.col("close") + eps).alias("f_range_pct"),
        )
        .drop(["_top", "_bottom"])
    )
