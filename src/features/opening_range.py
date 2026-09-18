from __future__ import annotations

import polars as pl


def add_opening_range_features(
    df: pl.DataFrame, windows: tuple[int, ...] = (1, 3, 6)
) -> pl.DataFrame:
    # Window sizes are 5m bars. Each feature is derived only from bars already closed.
    out = df
    for bars in windows:
        high = pl.col("high").shift(1).rolling_max(bars)
        low = pl.col("low").shift(1).rolling_min(bars)
        opening = (
            pl.col("close").first().over("session_date").alias(f"_session_open_close_{bars}")
        )
        out = out.with_columns(
            [
                high.over("session_date").alias(f"prior_{bars}bar_high"),
                low.over("session_date").alias(f"prior_{bars}bar_low"),
                opening,
            ]
        ).with_columns(
            [
                (
                    (pl.col("close") - pl.col(f"prior_{bars}bar_low"))
                    / (
                        pl.col(f"prior_{bars}bar_high")
                        - pl.col(f"prior_{bars}bar_low")
                        + 1e-9
                    )
                ).alias(f"position_in_prior_{bars}bar_range"),
                (
                    pl.col(f"prior_{bars}bar_high") - pl.col(f"prior_{bars}bar_low")
                ).alias(f"prior_{bars}bar_range"),
                (
                    (pl.col("close") - pl.col(f"_session_open_close_{bars}"))
                    / (pl.col(f"_session_open_close_{bars}").abs() + 1e-9)
                ).alias("session_return_from_open"),
            ]
        )
    return out.drop([c for c in out.columns if c.startswith("_session_open_close_")])
