from __future__ import annotations

import math

import polars as pl


def add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    # timestamp is the 5m candle close time. Market session is 09:15-15:30 IST.
    minutes = (
        (pl.col("timestamp").dt.hour() * 60 + pl.col("timestamp").dt.minute()) - 9 * 60 - 15
    )
    session_minutes = pl.lit(375)
    frac = minutes.cast(pl.Float64) / session_minutes
    tod_angle = frac * (2.0 * math.pi)
    weekday = pl.col("timestamp").dt.weekday()
    return df.with_columns(
        [
            minutes.clip(lower_bound=0, upper_bound=375).alias("minutes_from_open"),
            (375 - minutes).clip(lower_bound=0, upper_bound=375).alias("minutes_to_close"),
            pl.when(minutes < 30).then(1).otherwise(0).alias("is_opening_30m"),
            pl.when(minutes >= 330).then(1).otherwise(0).alias("is_closing_45m"),
            (tod_angle.sin()).alias("time_sin"),
            (tod_angle.cos()).alias("time_cos"),
            ((weekday - 1) / 4.0 * (2.0 * math.pi)).sin().alias("weekday_sin"),
            ((weekday - 1) / 4.0 * (2.0 * math.pi)).cos().alias("weekday_cos"),
        ]
    )
