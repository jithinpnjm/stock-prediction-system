from __future__ import annotations

import polars as pl


def add_opening_range_features(
    df: pl.DataFrame,
    windows: tuple[int, ...] = (1, 3, 6),
) -> pl.DataFrame:
    out = df
    minutes = (
        pl.col("timestamp").dt.hour() * 60 + pl.col("timestamp").dt.minute()
        - (9 * 60 + 15)
    )
    for bars in windows:
        window_minutes = bars * 5
        high = (
            pl.when(minutes <= window_minutes)
            .then(pl.col("high"))
            .otherwise(None)
            .max()
            .over("session_date")
        )
        low = (
            pl.when(minutes <= window_minutes)
            .then(pl.col("low"))
            .otherwise(None)
            .min()
            .over("session_date")
        )
        out = out.with_columns(
            [
                high.alias(f"opening_range_high_{bars}"),
                low.alias(f"opening_range_low_{bars}"),
            ]
        ).with_columns(
            [
                (
                    pl.col("opening_range_high_{bars}".format(bars=bars))
                    - pl.col("opening_range_low_{bars}".format(bars=bars))
                ).alias(f"opening_range_{bars}_size"),
                (
                    (
                        pl.col("close")
                        - pl.col(f"opening_range_low_{bars}")
                    )
                    / (
                        pl.col(f"opening_range_high_{bars}")
                        - pl.col(f"opening_range_low_{bars}")
                        + 1e-9
                    )
                ).alias(f"position_in_opening_range_{bars}"),
                (
                    pl.col("close") > pl.col(f"opening_range_high_{bars}")
                ).cast(pl.Int8).alias(f"breaks_opening_high_{bars}"),
                (
                    pl.col("close") < pl.col(f"opening_range_low_{bars}")
                ).cast(pl.Int8).alias(f"breaks_opening_low_{bars}"),
            ]
        )
    return out
