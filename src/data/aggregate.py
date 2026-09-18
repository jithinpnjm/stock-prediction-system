from __future__ import annotations

import polars as pl

from src.common.contracts import MARKET_TIMEZONE, SESSION_OPEN


def aggregate_1m_to_5m(
    df_1m: pl.DataFrame,
    *,
    drop_incomplete: bool = True,
) -> pl.DataFrame:
    if df_1m.is_empty():
        raise ValueError("Cannot aggregate empty 1m data")

    minute_of_day = (
        pl.col("timestamp").dt.hour() * 60 + pl.col("timestamp").dt.minute()
    )
    session_open_minutes = SESSION_OPEN.hour * 60 + SESSION_OPEN.minute
    bucket_index = (
        ((minute_of_day - session_open_minutes) / 5.0)
        .floor()
        .cast(pl.Int64)
    )

    out = (
        df_1m.sort("timestamp")
        .with_columns(
            [
                bucket_index.alias("_bucket_index"),
                pl.col("timestamp").dt.date().alias("session_date"),
            ]
        )
        .filter(pl.col("_bucket_index").is_between(0, 74))
        .group_by(["session_date", "_bucket_index"], maintain_order=True)
        .agg(
            [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.len().alias("source_1m_count"),
            ]
        )
        .with_columns(
            (
                pl.col("session_date").cast(
                    pl.Datetime(time_zone=MARKET_TIMEZONE)
                )
                + pl.duration(minutes=session_open_minutes)
                + pl.duration(minutes=5) * (pl.col("_bucket_index") + 1)
            ).alias("timestamp")
        )
        .select(
            [
                "session_date",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "source_1m_count",
            ]
        )
        .sort("timestamp")
    )

    if drop_incomplete:
        out = out.filter(pl.col("source_1m_count") == 5)
    if out.is_empty():
        raise ValueError("No complete 5m buckets after aggregation")
    return out


def validate_aggregation(df_5m: pl.DataFrame) -> None:
    if df_5m.is_empty():
        raise ValueError("5m dataset is empty")

    invalid = df_5m.filter(
        (pl.col("high") < pl.col("low"))
        | (pl.col("open") < pl.col("low"))
        | (pl.col("open") > pl.col("high"))
        | (pl.col("close") < pl.col("low"))
        | (pl.col("close") > pl.col("high"))
    )
    if invalid.height:
        raise ValueError(f"Invalid 5m OHLC rows: {invalid.height}")

    counts = df_5m.group_by("session_date").agg(pl.len().alias("count"))
    if counts.filter(pl.col("count") != 75).height:
        raise ValueError("Each validated 5m session must contain exactly 75 bars")
