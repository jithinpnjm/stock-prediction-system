from __future__ import annotations

import polars as pl


def add_multi_timeframe_features(
    df: pl.DataFrame, timeframes: tuple[int, ...] = (15, 30, 60)
) -> pl.DataFrame:
    out = df
    for minutes in timeframes:
        bars = minutes // 5
        prev_close = pl.col("close").shift(bars)
        prior_high = pl.col("high").shift(1).rolling_max(bars)
        prior_low = pl.col("low").shift(1).rolling_min(bars)
        prior_volume = pl.col("volume").shift(1).rolling_sum(bars)
        out = out.with_columns(
            [
                (
                    pl.col("close") / prev_close - 1.0
                ).over("session_date").alias(f"htf_{minutes}m_return"),
                (
                    (prior_high - prior_low) / (pl.col("close").abs() + 1e-9) * 10_000
                ).over("session_date").alias(f"htf_{minutes}m_range_bps"),
                (
                    pl.col("close") / prior_high - 1.0
                ).over("session_date").alias(f"htf_{minutes}m_vs_prior_high"),
                (
                    pl.col("close") / prior_low - 1.0
                ).over("session_date").alias(f"htf_{minutes}m_vs_prior_low"),
                (
                    pl.col("volume") / (prior_volume / max(bars, 1) + 1e-9)
                ).over("session_date").alias(f"htf_{minutes}m_volume_ratio"),
            ]
        )
    return out
