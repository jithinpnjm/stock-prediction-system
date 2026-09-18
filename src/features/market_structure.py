from __future__ import annotations

import polars as pl


def add_causal_market_structure(
    df: pl.DataFrame,
    lookback: int = 5,
) -> pl.DataFrame:
    if lookback < 2:
        raise ValueError("lookback must be >= 2")

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

    out = df.with_columns(
        [
            prior_high.alias("prior_swing_high"),
            prior_low.alias("prior_swing_low"),
            (
                pl.col("close") > prior_high
            ).cast(pl.Int8).alias(
                "breaks_prior_high"
            ),
            (
                pl.col("close") < prior_low
            ).cast(pl.Int8).alias(
                "breaks_prior_low"
            ),
        ]
    ).with_columns(
        [
            (
                (pl.col("close") - prior_high)
                / (pl.col("atr_14") + 1e-9)
            ).alias("distance_to_prior_high_atr"),
            (
                (pl.col("close") - prior_low)
                / (pl.col("atr_14") + 1e-9)
            ).alias("distance_to_prior_low_atr"),
            pl.when(
                pl.col("breaks_prior_high") == 1
            )
            .then(1)
            .when(
                pl.col("breaks_prior_low") == 1
            )
            .then(-1)
            .otherwise(0)
            .alias("_break_direction"),
        ]
    ).with_columns(
        [
            pl.col("_break_direction")
            .shift(1)
            .forward_fill()
            .over("session_date")
            .alias("_prior_break_direction"),
        ]
    ).with_columns(
        [
            (
                (pl.col("_break_direction") == 1)
                & (pl.col("_prior_break_direction") == -1)
            ).cast(pl.Int8).alias(
                "choch_up_candidate"
            ),
            (
                (pl.col("_break_direction") == -1)
                & (pl.col("_prior_break_direction") == 1)
            ).cast(pl.Int8).alias(
                "choch_down_candidate"
            ),
            pl.col("_break_direction").alias(
                "structure_break_direction"
            ),
        ]
    ).drop(
        ["_break_direction", "_prior_break_direction"]
    )

    return out
