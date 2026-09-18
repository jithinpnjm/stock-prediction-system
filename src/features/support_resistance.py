from __future__ import annotations

import polars as pl


def add_support_resistance_features(
    df: pl.DataFrame,
    windows: tuple[int, ...] = (12, 36, 75),
    touch_atr_fraction: float = 0.25,
) -> pl.DataFrame:
    if touch_atr_fraction <= 0:
        raise ValueError(
            "touch_atr_fraction must be > 0"
        )

    out = df

    for window in windows:
        resistance = (
            pl.col("high")
            .shift(1)
            .rolling_max(window)
            .over("session_date")
        )
        support = (
            pl.col("low")
            .shift(1)
            .rolling_min(window)
            .over("session_date")
        )

        resistance_touch = (
            (
                (pl.col("close") - resistance).abs()
                <= touch_atr_fraction
                * pl.col("atr_14")
            )
            .cast(pl.Int8)
        )
        support_touch = (
            (
                (pl.col("close") - support).abs()
                <= touch_atr_fraction
                * pl.col("atr_14")
            )
            .cast(pl.Int8)
        )

        out = out.with_columns(
            [
                resistance.alias(
                    f"resistance_high_{window}"
                ),
                support.alias(
                    f"support_low_{window}"
                ),
                (
                    (pl.col("close") - resistance)
                    / (pl.col("atr_14") + 1e-9)
                ).alias(
                    f"distance_resistance_{window}_atr"
                ),
                (
                    (pl.col("close") - support)
                    / (pl.col("atr_14") + 1e-9)
                ).alias(
                    f"distance_support_{window}_atr"
                ),
                resistance_touch.alias(
                    f"resistance_touch_{window}"
                ),
                support_touch.alias(
                    f"support_touch_{window}"
                ),
                resistance_touch
                .rolling_sum(20)
                .over("session_date")
                .alias(
                    f"resistance_touch_count_{window}"
                ),
                support_touch
                .rolling_sum(20)
                .over("session_date")
                .alias(
                    f"support_touch_count_{window}"
                ),
            ]
        )

    return out
