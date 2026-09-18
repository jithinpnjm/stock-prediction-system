from __future__ import annotations

import math

import polars as pl


def add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    out = df.sort("timestamp")
    local = pl.col("timestamp").dt.convert_time_zone("Asia/Kolkata")
    minutes = local.dt.hour() * 60 + local.dt.minute()
    session_minutes = minutes - 9 * 60 - 15
    bars = session_minutes // 5
    phase = pl.when(session_minutes < 60).then(pl.lit(0)).when(
        session_minutes < 180
    ).then(pl.lit(1)).when(session_minutes < 300).then(pl.lit(2)).otherwise(pl.lit(3))
    minute_of_day = minutes
    return out.with_columns(
        local.dt.hour().alias("f_hour"),
        local.dt.minute().alias("f_minute"),
        local.dt.weekday().alias("f_weekday"),
        session_minutes.clip(lower_bound=0).alias("f_minutes_from_open"),
        (15 * 60 + 30 - minutes).clip(lower_bound=0).alias("f_minutes_to_close"),
        bars.clip(lower_bound=0).alias("f_session_bar_index"),
        phase.alias("f_session_phase"),
        (pl.lit(2 * math.pi) * minute_of_day / (24 * 60)).sin().alias("f_time_sin"),
        (pl.lit(2 * math.pi) * minute_of_day / (24 * 60)).cos().alias("f_time_cos"),
    )
