from __future__ import annotations

import polars as pl


def add_support_resistance_features(
    df: pl.DataFrame, windows: tuple[int, ...] = (12, 36, 75)
) -> pl.DataFrame:
    out = df
    for w in windows:
        out = out.with_columns(
            [
                pl.col("high").shift(1).rolling_max(w).alias(f"resistance_high_{w}"),
                pl.col("low").shift(1).rolling_min(w).alias(f"support_low_{w}"),
            ]
        ).with_columns(
            [
                (
                    (pl.col("close") - pl.col(f"resistance_high_{w}"))
                    / (pl.col("atr_14") + 1e-9)
                ).alias(f"distance_resistance_{w}_atr"),
                (
                    (pl.col("close") - pl.col(f"support_low_{w}"))
                    / (pl.col("atr_14") + 1e-9)
                ).alias(f"distance_support_{w}_atr"),
            ]
        )
    return out
