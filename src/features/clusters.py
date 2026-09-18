from __future__ import annotations

import polars as pl


def add_consecutive_cluster_features(df: pl.DataFrame, max_length: int = 10) -> pl.DataFrame:
    reset = (
        (pl.col("session_date") != pl.col("session_date").shift(1))
        | (pl.col("direction") != pl.col("direction").shift(1))
    ).fill_null(True)
    out = df.with_columns(reset.cast(pl.Int64).cum_sum().alias("_cluster_id"))
    out = out.with_columns(
        pl.col("direction").cum_count().over("_cluster_id").alias("_cluster_len"),
        pl.col("body").cum_sum().over("_cluster_id").alias("_cluster_body"),
        pl.col("range").cum_sum().over("_cluster_id").alias("_cluster_range"),
        pl.col("volume").cum_sum().over("_cluster_id").alias("_cluster_volume"),
    ).with_columns(
        pl.col("_cluster_len").clip(upper_bound=max_length).alias("consecutive_length"),
        (pl.col("_cluster_body") / (pl.col("close").abs() + 1e-9) * 10_000).alias("cluster_body_bps"),
        (pl.col("_cluster_range") / (pl.col("close").abs() + 1e-9) * 10_000).alias("cluster_range_bps"),
        (pl.col("_cluster_volume") / (pl.col("volume").rolling_mean(20).clip(lower_bound=1.0) * pl.col("_cluster_len"))).alias("cluster_volume_vs_avg"),
    )
    for n in range(2, max_length + 1):
        out = out.with_columns((pl.col("_cluster_len") >= n).cast(pl.Int8).alias(f"has_cluster_{n}"))
    return out.drop(["_cluster_id", "_cluster_len", "_cluster_body", "_cluster_range", "_cluster_volume"])
