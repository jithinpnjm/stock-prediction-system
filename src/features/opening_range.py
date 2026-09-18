from __future__ import annotations

import polars as pl


def add_opening_range_features(
    df: pl.DataFrame,
    windows: tuple[int, ...] = (1, 3, 6),
) -> pl.DataFrame:
    minutes_from_open = (
        pl.col("timestamp").dt.hour() * 60
        + pl.col("timestamp").dt.minute()
        - (9 * 60 + 15)
    )

    # 5m timestamps represent candle CLOSES:
    # 09:20 -> first bar, 09:25 -> second bar, ...
    session_bar_index = (
        (minutes_from_open / 5).floor()
        .cast(pl.Int64)
        - 1
    )

    out = df.with_columns(
        session_bar_index.alias("_session_bar_index")
    )

    for bars in windows:
        opening_high = (
            pl.when(
                pl.col("_session_bar_index") < bars
            )
            .then(pl.col("high"))
            .otherwise(None)
            .max()
            .over("session_date")
        )
        opening_low = (
            pl.when(
                pl.col("_session_bar_index") < bars
            )
            .then(pl.col("low"))
            .otherwise(None)
            .min()
            .over("session_date")
        )

        range_complete = (
            pl.col("_session_bar_index") >= bars - 1
        )
        breakout_window = (
            pl.col("_session_bar_index") >= bars
        )

        out = out.with_columns(
            [
                opening_high.alias(
                    f"opening_range_high_{bars}"
                ),
                opening_low.alias(
                    f"opening_range_low_{bars}"
                ),
                range_complete.cast(pl.Int8).alias(
                    f"opening_range_complete_{bars}"
                ),
            ]
        ).with_columns(
            [
                (
                    pl.col(f"opening_range_high_{bars}")
                    - pl.col(f"opening_range_low_{bars}")
                ).alias(
                    f"opening_range_{bars}_size"
                ),
                pl.when(range_complete)
                .then(
                    (
                        pl.col("close")
                        - pl.col(
                            f"opening_range_low_{bars}"
                        )
                    )
                    / (
                        pl.col(
                            f"opening_range_high_{bars}"
                        )
                        - pl.col(
                            f"opening_range_low_{bars}"
                        )
                        + 1e-9
                    )
                )
                .otherwise(None)
                .alias(
                    f"position_in_opening_range_{bars}"
                ),
                pl.when(breakout_window)
                .then(
                    (
                        pl.col("close")
                        > pl.col(
                            f"opening_range_high_{bars}"
                        )
                    ).cast(pl.Int8)
                )
                .otherwise(0)
                .alias(
                    f"breaks_opening_high_{bars}"
                ),
                pl.when(breakout_window)
                .then(
                    (
                        pl.col("close")
                        < pl.col(
                            f"opening_range_low_{bars}"
                        )
                    ).cast(pl.Int8)
                )
                .otherwise(0)
                .alias(
                    f"breaks_opening_low_{bars}"
                ),
            ]
        )

    return out.drop("_session_bar_index")
