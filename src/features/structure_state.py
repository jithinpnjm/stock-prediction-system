from __future__ import annotations

import polars as pl


def add_structure_state_features(
    df: pl.DataFrame,
    *,
    lookback: int = 5,
) -> pl.DataFrame:
    """
    Causal structure proxies.

    This intentionally does not label future-confirmed HH/HL/LH/LL states.
    It exposes only structure observable from bars at or before the current
    5m close.
    """
    prior_high = (
        pl.col("high")
        .shift(1)
        .rolling_max(lookback)
        .over("session_date")
    )
    prior_low = (
        pl.col("low")
        .shift(1)
        .rolling_min(lookback)
        .over("session_date")
    )

    return df.with_columns(
        [
            (
                pl.col("close") > prior_high
            ).cast(pl.Int8).alias("structure_break_up"),
            (
                pl.col("close") < prior_low
            ).cast(pl.Int8).alias("structure_break_down"),
            (
                (pl.col("close") - pl.col("open"))
                / (pl.col("atr_14") + 1e-9)
            ).alias("structure_body_atr"),
            (
                (pl.col("high") - pl.col("low"))
                / (pl.col("atr_14") + 1e-9)
            ).alias("structure_range_atr"),
        ]
    )
