from __future__ import annotations

import math
import polars as pl


def add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    out = df.sort("timestamp")
    local = pl.col("timestamp").dt.convert_time_zone("Asia/Kolkata")
    minutes = local.dt.hour().cast(pl.Int32) * 60 + local.dt.minute().cast(pl.Int32)
    session_minutes = minutes - 9 * 60 - 15
    # 5m close timestamps are 09:20, 09:25, ... 15:30.
    bar_index = ((session_minutes // 5) - 1).clip(lower_bound=0)
    phase = (
        pl.when(session_minutes < 60)
        .then(0)
        .when(session_minutes < 180)
        .then(1)
        .when(session_minutes < 300)
        .then(2)
        .otherwise(3)
    )
    angle = pl.lit(2 * math.pi) * minutes / (24 * 60)
    return out.with_columns(
        local.dt.hour().alias("f_hour"),
        local.dt.minute().alias("f_minute"),
        local.dt.weekday().alias("f_weekday"),
        session_minutes.clip(lower_bound=0).alias("f_minutes_from_open"),
        (15 * 60 + 30 - minutes).clip(lower_bound=0).alias("f_minutes_to_close"),
        bar_index.alias("f_session_bar_index"),
        phase.alias("f_session_phase"),
        angle.sin().alias("f_time_sin"),
        angle.cos().alias("f_time_cos"),
    )
