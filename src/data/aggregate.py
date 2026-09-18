from __future__ import annotations

import polars as pl

from .calendar import NSECalendar


def aggregate_1m_to_5m(
    df_1m: pl.DataFrame,
    *,
    calendar: NSECalendar,
    drop_incomplete: bool = True,
) -> pl.DataFrame:
    if df_1m.is_empty():
        return df_1m

    df = calendar.filter_session(df_1m).sort("timestamp")
    local = pl.col("timestamp").dt.convert_time_zone(calendar.timezone)
    session_minute = (
        local.dt.hour().cast(pl.Int32) * 60
        + local.dt.minute().cast(pl.Int32)
        - (calendar.session_open.hour * 60 + calendar.session_open.minute)
    )
    # Canonical 1m close timestamps are 09:16..15:30. Bucket boundaries are
    # 09:20, 09:25, ... 15:30.
    df = df.with_columns(
        (((session_minute + 4) / 5).floor().cast(pl.Int32)).alias("_bucket"),
        pl.col("timestamp").dt.date().alias("_session_date"),
    )

    out = (
        df.group_by(["_session_date", "_bucket"], maintain_order=True)
        .agg(
            pl.col("timestamp").min().alias("_first_ts"),
            pl.col("timestamp").max().alias("_last_ts"),
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.len().alias("_n_1m"),
        )
        .sort(["_session_date", "_bucket"])
    )
    if drop_incomplete:
        out = out.filter(pl.col("_n_1m") == 5)
    return out.select(
        ["_last_ts", "_session_date", "open", "high", "low", "close", "volume", "_n_1m"]
    ).rename({"_last_ts": "timestamp", "_session_date": "session_date", "_n_1m": "source_1m_count"})


def validate_aggregation(df_5m: pl.DataFrame, calendar: NSECalendar | None = None) -> None:
    if df_5m.is_empty():
        raise ValueError("5m dataset is empty")
    bad = df_5m.filter(
        (pl.col("high") < pl.col("low"))
        | (pl.col("open") < pl.col("low"))
        | (pl.col("open") > pl.col("high"))
        | (pl.col("close") < pl.col("low"))
        | (pl.col("close") > pl.col("high"))
    )
    if bad.height:
        raise ValueError(f"invalid 5m OHLC geometry: {bad.height}")
    if calendar is None:
        return
    for d in df_5m.select(pl.col("timestamp").dt.date().unique()).to_series().to_list():
        day = df_5m.filter(pl.col("timestamp").dt.date() == d).sort("timestamp")
        if day.height != 75:
            raise ValueError(f"{d}: expected 75 complete 5m bars, found {day.height}")
        if day["timestamp"][0].strftime("%H:%M") != "09:20":
            raise ValueError(f"{d}: first timestamp must be 09:20")
        if day["timestamp"][-1].strftime("%H:%M") != "15:30":
            raise ValueError(f"{d}: last timestamp must be 15:30")
        deltas = day.with_columns(pl.col("timestamp").diff().dt.total_seconds().alias("_d")).filter(
            pl.col("_d").is_not_null()
        )
        if deltas.filter(pl.col("_d") != 300).height:
            raise ValueError(f"{d}: canonical 5m timestamps are not exactly 5 minutes apart")
