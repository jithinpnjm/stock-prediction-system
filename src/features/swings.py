from __future__ import annotations

import polars as pl


def add_swing_candidates(df: pl.DataFrame, lookback: int = 5) -> pl.DataFrame:
    prev_high = pl.col("high").shift(1).rolling_max(lookback)
    prev_low = pl.col("low").shift(1).rolling_min(lookback)
    return df.with_columns(
        [
            (pl.col("high") >= prev_high).cast(pl.Int8).alias("causal_swing_high"),
            (pl.col("low") <= prev_low).cast(pl.Int8).alias("causal_swing_low"),
        ]
    )
