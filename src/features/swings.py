from __future__ import annotations

import polars as pl


def add_swing_features(
    df: pl.DataFrame,
    lookback: int = 3,
) -> pl.DataFrame:
    """Causal swing candidates using only current and prior bars."""
    out = df.sort("timestamp")
    prior_high = pl.col("high").shift(1).rolling_max(lookback)
    prior_low = pl.col("low").shift(1).rolling_min(lookback)
    prior_close = pl.col("close").shift(1)
    return out.with_columns(
        pl.when(pl.col("high") >= prior_high).then(1).otherwise(0).alias(
            "f_swing_high_candidate"
        ),
        pl.when(pl.col("low") <= prior_low).then(1).otherwise(0).alias(
            "f_swing_low_candidate"
        ),
        (pl.col("high") - prior_high).alias("f_breakout_above_prior_high"),
        (prior_low - pl.col("low")).alias("f_breakdown_below_prior_low"),
        (pl.col("close") - prior_close).alias("f_structure_return"),
    )
