from __future__ import annotations

import polars as pl


def add_consecutive_cluster_features(
    df: pl.DataFrame, max_length: int = 10
) -> pl.DataFrame:
    direction = pl.col("direction") if "direction" in df.columns else (
        pl.when(pl.col("close") > pl.col("open")).then(1)
        .when(pl.col("close") < pl.col("open")).then(-1)
        .otherwise(0)
    )
    out = df.with_columns(direction.alias("_direction"))

    # A cluster is the consecutive same-direction run ending at the current bar.
    reset = (pl.col("_direction") != pl.col("_direction").shift(1)).fill_null(True)
    out = out.with_columns(reset.cast(pl.Int64).cum_sum().alias("_cluster_id"))

    stats = (
        out.group_by("_cluster_id", maintain_order=True)
        .agg(
            [
                pl.len().alias("_cluster_len"),
                pl.col("body").sum().alias("_cluster_body"),
                pl.col("range").sum().alias("_cluster_range"),
                pl.col("volume").sum().alias("_cluster_volume"),
                pl.col("upper_wick_pct_range").mean().alias("_cluster_upper_wick_mean"),
                pl.col("lower_wick_pct_range").mean().alias("_cluster_lower_wick_mean"),
            ]
        )
    )

    out = out.join(stats, on="_cluster_id", how="left")
    # Cap the exposed streak length while retaining cumulative statistics.
    out = out.with_columns(
        [
            pl.col("_cluster_len").clip(upper_bound=max_length).alias("consecutive_length"),
            (pl.col("_cluster_body") / (pl.col("close").abs() + 1e-9) * 10_000).alias(
                "cluster_body_bps"
            ),
            (pl.col("_cluster_range") / (pl.col("close").abs() + 1e-9) * 10_000).alias(
                "cluster_range_bps"
            ),
            (pl.col("_cluster_volume") / pl.col("volume").rolling_mean(20).clip(lower_bound=1.0)).alias(
                "cluster_volume_vs_avg"
            ),
        ]
    )
    for n in range(2, max_length + 1):
        out = out.with_columns(
            pl.when(pl.col("_cluster_len") >= n).then(1).otherwise(0).alias(f"has_cluster_{n}")
        )

    return out.drop(
        [
            "_direction",
            "_cluster_id",
            "_cluster_len",
            "_cluster_body",
            "_cluster_range",
            "_cluster_volume",
            "_cluster_upper_wick_mean",
            "_cluster_lower_wick_mean",
        ]
    )
