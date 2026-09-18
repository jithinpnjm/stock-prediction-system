from __future__ import annotations

import polars as pl


def add_swing_candidates(
    df: pl.DataFrame,
    *,
    confirmation_bars: int = 2,
) -> pl.DataFrame:
    if confirmation_bars < 1:
        raise ValueError(
            "confirmation_bars must be >= 1"
        )

    k = confirmation_bars

    high_candidate = (
        pl.col("high")
        .shift(k)
        .over("session_date")
    )
    low_candidate = (
        pl.col("low")
        .shift(k)
        .over("session_date")
    )

    left_high = (
        pl.col("high")
        .shift(k + 1)
        .rolling_max(k)
        .over("session_date")
    )
    right_high = (
        pl.col("high")
        .shift(1)
        .rolling_max(max(k - 1, 1))
        .over("session_date")
    )
    left_low = (
        pl.col("low")
        .shift(k + 1)
        .rolling_min(k)
        .over("session_date")
    )
    right_low = (
        pl.col("low")
        .shift(1)
        .rolling_min(max(k - 1, 1))
        .over("session_date")
    )

    confirmed_high = (
        high_candidate >= left_high
    ) & (
        high_candidate >= right_high
    )
    confirmed_low = (
        low_candidate <= left_low
    ) & (
        low_candidate <= right_low
    )

    out = df.with_columns(
        [
            confirmed_high.cast(pl.Int8).alias(
                "causal_swing_high_confirmed"
            ),
            confirmed_low.cast(pl.Int8).alias(
                "causal_swing_low_confirmed"
            ),
            pl.when(confirmed_high)
            .then(high_candidate)
            .otherwise(None)
            .alias("_confirmed_high_value"),
            pl.when(confirmed_low)
            .then(low_candidate)
            .otherwise(None)
            .alias("_confirmed_low_value"),
        ]
    ).with_columns(
        [
            pl.col("_confirmed_high_value")
            .shift(1)
            .forward_fill()
            .over("session_date")
            .alias("causal_swing_high"),
            pl.col("_confirmed_low_value")
            .shift(1)
            .forward_fill()
            .over("session_date")
            .alias("causal_swing_low"),
        ]
    ).with_columns(
        [
            (
                pl.col("causal_swing_high")
                > pl.col("causal_swing_high")
                .shift(1)
                .over("session_date")
            )
            .cast(pl.Int8)
            .alias("causal_swing_higher_high"),
            (
                pl.col("causal_swing_low")
                > pl.col("causal_swing_low")
                .shift(1)
                .over("session_date")
            )
            .cast(pl.Int8)
            .alias("causal_swing_higher_low"),
            (
                pl.col("causal_swing_high")
                < pl.col("causal_swing_high")
                .shift(1)
                .over("session_date")
            )
            .cast(pl.Int8)
            .alias("causal_swing_lower_high"),
            (
                pl.col("causal_swing_low")
                < pl.col("causal_swing_low")
                .shift(1)
                .over("session_date")
            )
            .cast(pl.Int8)
            .alias("causal_swing_lower_low"),
            (
                (
                    pl.col("close")
                    - pl.col("causal_swing_high")
                )
                / (pl.col("atr_14") + 1e-9)
            ).alias("distance_causal_swing_high_atr"),
            (
                (
                    pl.col("close")
                    - pl.col("causal_swing_low")
                )
                / (pl.col("atr_14") + 1e-9)
            ).alias("distance_causal_swing_low_atr"),
        ]
    ).drop(
        ["_confirmed_high_value", "_confirmed_low_value"]
    )

    return out
