from __future__ import annotations

import polars as pl


def add_consecutive_cluster_features(
    df: pl.DataFrame, max_length: int = 10
) -> pl.DataFrame:
    out = df.with_columns(
        [
            (pl.col("direction") != pl.col("direction").shift(1))
            .fill_null(True)
            .cast(pl.Int64)
            .cum_sum()
            .alias("_cluster_id"),
        ]
    ).with_columns(
        [
            pl.col("direction").cum_count().over("_cluster_id").alias("_cluster_len"),
            pl.col("body").cum_sum().over("_cluster_id").alias("_cluster_body"),
            pl.col("range").cum_sum().over("_cluster_id").alias("_cluster_range"),
            pl.col("volume").cum_sum().over("_cluster_id").alias("_cluster_volume"),
            pl.col("upper_wick_pct_range").cum_mean().over("_cluster_id").alias("_cluster_upper_wick_mean"),
            pl.col("lower_wick_pct_range").cum_mean().over("_cluster_id").alias("_cluster_lower_wick_mean"),
        ]
    ).with_columns(
        [
            pl.col("_cluster_len").clip(upper_bound=max_length).alias("consecutive_length"),
            (
                pl.col("_cluster_body") / (pl.col("close").abs() + 1e-9) * 10_000
            ).alias("cluster_body_bps"),
            (
                pl.col("_cluster_range") / (pl.col("close").abs() + 1e-9) * 10_000
            ).alias("cluster_range_bps"),
            (
                pl.col("_cluster_volume")
                / (
                    pl.col("volume")
                    .rolling_mean(20)
                    .clip(lower_bound=1.0)
                    * pl.col("_cluster_len")
                )
            ).alias("cluster_volume_vs_avg"),
        ]
    )

    for n in range(2, max_length + 1):
        out = out.with_columns(
            (pl.col("_cluster_len") >= n).cast(pl.Int8).alias(f"has_cluster_{n}")
        )
    return out.drop(
        [
            "_cluster_id",
            "_cluster_len",
            "_cluster_body",
            "_cluster_range",
            "_cluster_volume",
            "_cluster_upper_wick_mean",
            "_cluster_lower_wick_mean",
        ]
    )
