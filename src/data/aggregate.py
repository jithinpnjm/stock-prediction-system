from __future__ import annotations

import polars as pl

from .calendar import NSECalendar

_BUCKET_MINUTES = 5


def aggregate_1m_to_5m(
    df_1m: pl.DataFrame,
    *,
    calendar: NSECalendar,
    drop_incomplete: bool = True,
) -> pl.DataFrame:
    """Aggregate session-contained 1m candles into close-timestamped 5m bars.

    The upstream contract defines input timestamps as 1-minute candle close
    timestamps. Each output bar retains the last 1m close timestamp so feature
    availability is unambiguous.
    """
    if df_1m.is_empty():
        return df_1m

    df = calendar.filter_session(df_1m).sort("timestamp")
    local = pl.col("timestamp").dt.convert_time_zone(calendar.timezone)
    session_minutes = (
        local.dt.hour() * 60
        + local.dt.minute()
        - (calendar.session_open.hour * 60 + calendar.session_open.minute)
    )
    df = df.with_columns(
        (session_minutes // _BUCKET_MINUTES).alias("_bucket"),
        pl.col("timestamp").dt.date().alias("_session_date"),
    )

    out = (
        df.group_by(["_session_date", "_bucket"], maintain_order=True)
        .agg(
            [
                pl.col("timestamp").min().alias("_first_ts"),
                pl.col("timestamp").max().alias("_last_ts"),
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.len().alias("_n_1m"),
            ]
        )
        .with_columns(pl.col("_last_ts").alias("timestamp"))
        .sort(["_session_date", "_bucket"])
    )

    if drop_incomplete:
        out = out.filter(pl.col("_n_1m") == _BUCKET_MINUTES)

    return out.select(["timestamp", "open", "high", "low", "close", "volume"])


def validate_aggregation(df_5m: pl.DataFrame) -> None:
    if df_5m.is_empty():
        raise ValueError("5m dataset is empty")

    invalid = df_5m.filter(
        (pl.col("high") < pl.col("low"))
        | (pl.col("open") < pl.col("low"))
        | (pl.col("open") > pl.col("high"))
        | (pl.col("close") < pl.col("low"))
        | (pl.col("close") > pl.col("high"))
        | (pl.col("high") - pl.col("low") < 0)
    )
    if invalid.height:
        raise ValueError(f"Invalid 5m OHLC geometry in {invalid.height} rows")
