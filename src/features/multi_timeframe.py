from __future__ import annotations

import polars as pl


def _completed_higher_tf(df: pl.DataFrame, minutes: int) -> pl.DataFrame:
    # Input timestamps are 5m close times. Generate higher-TF bars and label them
    # with their close timestamp. We join only completed higher-TF bars.
    if minutes % 5:
        raise ValueError("Higher timeframe must be a multiple of 5 minutes")
    return (
        df.group_by_dynamic(
            "timestamp",
            every=f"{minutes}m",
            period=f"{minutes}m",
            closed="right",
            label="right",
        )
        .agg(
            [
                pl.col("open").first().alias("htf_open"),
                pl.col("high").max().alias("htf_high"),
                pl.col("low").min().alias("htf_low"),
                pl.col("close").last().alias("htf_close"),
                pl.col("volume").sum().alias("htf_volume"),
                pl.len().alias("htf_count"),
            ]
        )
        .filter(pl.col("htf_count") == minutes // 5)
        .select(
            [
                pl.col("timestamp").alias("htf_timestamp"),
                "htf_open",
                "htf_high",
                "htf_low",
                "htf_close",
                "htf_volume",
            ]
        )
    )


def add_multi_timeframe_features(
    df: pl.DataFrame, timeframes: tuple[int, ...] = (15, 30, 60)
) -> pl.DataFrame:
    out = df
    for minutes in timeframes:
        htf = _completed_higher_tf(df.select(["timestamp", "open", "high", "low", "close", "volume"]), minutes)
        out = out.join_asof(
            htf.sort("htf_timestamp"),
            left_on="timestamp",
            right_on="htf_timestamp",
            strategy="backward",
        ).with_columns(
            [
                ((pl.col("htf_close") / pl.col("htf_open")) - 1.0).alias(
                    f"htf_{minutes}m_return"
                ),
                (
                    (pl.col("htf_high") - pl.col("htf_low"))
                    / (pl.col("htf_close").abs() + 1e-9)
                    * 10_000
                ).alias(f"htf_{minutes}m_range_bps"),
            ]
        ).drop(
            ["htf_open", "htf_high", "htf_low", "htf_close", "htf_volume"]
        )
    return out
