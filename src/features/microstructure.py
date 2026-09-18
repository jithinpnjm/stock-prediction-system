from __future__ import annotations

import polars as pl


def add_1m_inside_5m_features(
    one_minute: pl.DataFrame, five_minute: pl.DataFrame
) -> pl.DataFrame:
    # five_minute.timestamp is close time; 1m timestamp is start time.
    m = one_minute.with_columns(
        [
            pl.col("timestamp").dt.truncate("5m").alias("_bucket_start"),
            pl.col("close").diff().alias("_minute_delta"),
        ]
    )
    f = five_minute.with_columns(
        (
            pl.col("timestamp") - pl.duration(minutes=5)
        ).alias("_bucket_start")
    )

    agg = m.group_by("_bucket_start").agg(
        [
            pl.len().alias("minute_count"),
            pl.col("_minute_delta").abs().sum().alias("path_abs_move"),
            pl.col("_minute_delta").std().alias("minute_return_std"),
            (pl.col("close").last() - pl.col("close").first()).alias("first_last_move"),
            (pl.col("high").max() - pl.col("low").min()).alias("intrabar_range"),
            (pl.col("close").filter(pl.col("_minute_delta") > 0).count()).alias("up_minutes"),
            (pl.col("close").filter(pl.col("_minute_delta") < 0).count()).alias("down_minutes"),
            pl.col("volume").sum().alias("minute_volume"),
        ]
    )
    return f.join(agg, on="_bucket_start", how="left").drop("_bucket_start").with_columns(
        [
            (
                pl.col("up_minutes")
                / (pl.col("minute_count").clip(lower_bound=1))
            ).alias("up_minute_ratio"),
            (
                pl.col("path_abs_move")
                / (pl.col("intrabar_range").abs() + 1e-9)
            ).alias("path_efficiency"),
        ]
    )
