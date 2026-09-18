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

    # Canonical 1m availability timestamps are 09:16..15:30.
    # Five-minute buckets are aligned to the NSE 09:15 session boundary and
    # represented by their closing availability timestamp.
    bucket_close = local.dt.truncate("5m") + pl.duration(minutes=5)

    df = df.with_columns(
        bucket_close.alias("_bucket_close"),
        pl.col("timestamp").dt.date().alias("_session_date"),
    )

    out = (
        df.group_by(
            ["_session_date", "_bucket_close"],
            maintain_order=True,
        )
        .agg(
            pl.col("timestamp").min().alias("_first_ts"),
            pl.col("_bucket_close").first().alias("timestamp"),
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.len().alias("source_1m_count"),
        )
        .sort(["_session_date", "_bucket_close"])
    )

    if drop_incomplete:
        out = out.filter(pl.col("source_1m_count") == 5)

    return out.select(
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source_1m_count",
        ]
    )


def validate_aggregation(
    df_5m: pl.DataFrame,
    calendar: NSECalendar | None = None,
) -> None:
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

    if "source_1m_count" in df_5m.columns:
        incomplete = df_5m.filter(pl.col("source_1m_count") != 5)
        if incomplete.height:
            raise ValueError(
                f"5m dataset contains incomplete buckets: {incomplete.height}"
            )

    if calendar is None:
        return

    dates = (
        df_5m.select(pl.col("timestamp").dt.date().unique())
        .to_series()
        .to_list()
    )
    for d in dates:
        day = df_5m.filter(pl.col("timestamp").dt.date() == d).sort("timestamp")
        if day.height != 75:
            raise ValueError(
                f"{d}: expected 75 complete 5m bars, found {day.height}"
            )
        if day["timestamp"][0].strftime("%H:%M") != "09:20":
            raise ValueError(f"{d}: first timestamp must be 09:20")
        if day["timestamp"][-1].strftime("%H:%M") != "15:30":
            raise ValueError(f"{d}: last timestamp must be 15:30")

        deltas = (
            day.with_columns(
                pl.col("timestamp").diff().dt.total_seconds().alias("_d")
            )
            .filter(pl.col("_d").is_not_null())
        )
        if deltas.filter(pl.col("_d") != 300).height:
            raise ValueError(
                f"{d}: canonical 5m timestamps are not exactly 5 minutes apart"
            )
