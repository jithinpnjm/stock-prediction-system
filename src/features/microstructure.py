from __future__ import annotations

import polars as pl


def add_1m_inside_5m_features(
    df_1m: pl.DataFrame,
    df_5m: pl.DataFrame,
) -> pl.DataFrame:
    # Each 1m row belongs to exactly one 5m bucket (the bucket whose
    # close timestamp it is <= to, within the preceding 5 minutes).
    # Compute that bucket's close timestamp directly instead of joining
    # every 5m row against every 1m row of the same calendar date and
    # filtering afterwards -- that cross join is O(75 x 375) per day
    # and runs out of memory over multi-year datasets.
    bucket_close = (pl.col("timestamp") - pl.duration(minutes=1)).dt.truncate("5m") + pl.duration(
        minutes=5
    )
    one = df_1m.sort("timestamp").with_columns(
        bucket_close.alias("timestamp"),
        (pl.col("close") > pl.col("open")).cast(pl.Int8).alias("_up"),
        (pl.col("close") < pl.col("open")).cast(pl.Int8).alias("_down"),
        (pl.col("high") - pl.col("low")).alias("_range"),
    )
    five = df_5m.sort("timestamp")
    summary = (
        one.group_by("timestamp")
        .agg(
            pl.col("high").max().alias("_1m_path_high"),
            pl.col("low").min().alias("_1m_path_low"),
            pl.col("open").first().alias("_first_1m_open"),
            pl.col("close").last().alias("_last_1m_close"),
            # Not "f_"-prefixed: raw volume is 0 for the whole dataset
            # before 2025-07-01 (see clusters.py for the same note), so
            # this is degenerate for most of the 5-year history.
            pl.col("volume").sum().alias("_1m_path_volume"),
            pl.col("_range").sum().alias("f_1m_range_sum"),
            pl.col("_range").mean().alias("f_1m_range_mean"),
            pl.col("_range").std().alias("f_1m_range_std"),
            pl.col("_up").sum().alias("f_1m_up_count"),
            pl.col("_down").sum().alias("f_1m_down_count"),
            pl.len().alias("f_1m_count"),
        )
        .with_columns(
            (pl.col("_last_1m_close") - pl.col("_first_1m_open")).alias("f_1m_path_net_move"),
            (pl.col("_1m_path_high") - pl.col("_1m_path_low")).alias("f_1m_path_range"),
            (pl.col("_1m_path_high") - pl.col("_last_1m_close")).alias("f_1m_path_high_rejection"),
            (pl.col("_last_1m_close") - pl.col("_1m_path_low")).alias("f_1m_path_low_rebound"),
            (pl.col("f_1m_up_count") / (pl.col("f_1m_count") + 1e-9)).alias("f_1m_up_ratio"),
        )
        .drop(["_first_1m_open", "_last_1m_close", "_1m_path_high", "_1m_path_low"])
    )
    return five.join(summary, on="timestamp", how="left")
