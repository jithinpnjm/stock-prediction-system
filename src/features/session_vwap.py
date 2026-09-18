from __future__ import annotations

import polars as pl


def add_session_vwap(df: pl.DataFrame) -> pl.DataFrame:
    out = df.sort("timestamp").with_columns(
        pl.col("timestamp").dt.date().alias("_date"),
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias(
            "_typical_price"
        ),
    )
    return (
        out.with_columns(
            (pl.col("_typical_price") * pl.col("volume"))
            .cum_sum()
            .over("_date")
            .alias("_pv"),
            pl.col("volume").cum_sum().over("_date").alias("_v"),
        )
        .with_columns(
            (pl.col("_pv") / (pl.col("_v") + 1e-9)).alias("f_session_vwap"),
            (
                pl.col("close") - pl.col("_pv") / (pl.col("_v") + 1e-9)
            ).alias("f_vwap_distance"),
        )
        .drop(["_date", "_typical_price", "_pv", "_v"])
    )
