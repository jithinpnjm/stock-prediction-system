from __future__ import annotations

import polars as pl


def add_causal_market_structure(
    df: pl.DataFrame, lookback: int = 5
) -> pl.DataFrame:
    if lookback < 2:
        raise ValueError("lookback must be >= 2")

    # These are causal breakout/swing candidates: the current bar is compared
    # only with prior completed bars. No centered windows are used.
    prior_high = pl.col("high").shift(1).rolling_max(lookback)
    prior_low = pl.col("low").shift(1).rolling_min(lookback)
    out = df.with_columns(
        [
            prior_high.alias("prior_swing_high"),
            prior_low.alias("prior_swing_low"),
            (pl.col("close") > prior_high).cast(pl.Int8).alias("breaks_prior_high"),
            (pl.col("close") < prior_low).cast(pl.Int8).alias("breaks_prior_low"),
        ]
    )

    out = out.with_columns(
        [
            pl.col("high").shift(1).rolling_max(lookback).alias("_last_high_candidate"),
            pl.col("low").shift(1).rolling_min(lookback).alias("_last_low_candidate"),
        ]
    ).with_columns(
        [
            (
                (pl.col("close") - pl.col("_last_high_candidate"))
                / (pl.col("atr_14") + 1e-9)
            ).alias("distance_to_prior_high_atr"),
            (
                (pl.col("close") - pl.col("_last_low_candidate"))
                / (pl.col("atr_14") + 1e-9)
            ).alias("distance_to_prior_low_atr"),
        ]
    ).drop(["_last_high_candidate", "_last_low_candidate"])
    return out
