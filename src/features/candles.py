from __future__ import annotations

import polars as pl


def add_candle_geometry_features(df: pl.DataFrame) -> pl.DataFrame:
    eps = pl.lit(1e-9)
    return (
        df.with_columns(
            [
                (pl.col("high") - pl.col("low")).alias("range"),
                (pl.col("close") - pl.col("open")).alias("body"),
                pl.max_horizontal("open", "close").alias("_body_top"),
                pl.min_horizontal("open", "close").alias("_body_bottom"),
            ]
        )
        .with_columns(
            [
                (pl.col("high") - pl.col("_body_top")).alias("upper_wick"),
                (pl.col("_body_bottom") - pl.col("low")).alias("lower_wick"),
                (pl.col("body").abs() / (pl.col("range") + eps)).alias("body_pct_range"),
                (pl.col("upper_wick") / (pl.col("range") + eps)).alias("upper_wick_pct_range"),
                (pl.col("lower_wick") / (pl.col("range") + eps)).alias("lower_wick_pct_range"),
                (pl.col("range") / (pl.col("close").abs() + eps) * 10_000).alias("range_bps"),
                (pl.col("body") / (pl.col("close").shift(1).abs() + eps) * 10_000).alias("body_bps"),
                pl.when(pl.col("close") > pl.col("open"))
                .then(1)
                .when(pl.col("close") < pl.col("open"))
                .then(-1)
                .otherwise(0)
                .alias("direction"),
                ((pl.col("close") - pl.col("low")) / (pl.col("range") + eps)).alias("close_location"),
            ]
        )
        .drop(["_body_top", "_body_bottom"])
    )
