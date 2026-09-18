from __future__ import annotations

from datetime import timedelta

import polars as pl

from src.common.contracts import MARKET_TIMEZONE, SESSION_CLOSE, SESSION_OPEN
from src.data.calendar import expected_5m_close_timestamps


def aggregate_1m_to_5m(
    df_1m: pl.DataFrame,
    *,
    drop_incomplete: bool = True,
) -> pl.DataFrame:
    if df_1m.is_empty():
        raise ValueError("Cannot aggregate empty 1m data")
    df = df_1m.sort("timestamp")

    # Fyers-style source timestamps represent candle start. Build 5m windows
    # anchored at 09:15 and explicitly label each bucket by candle close.
    out = (
        df.with_columns(
            (
                pl.col("timestamp")
                - pl.col("timestamp").dt.truncate("1d")
                + pl.lit(SESSION_OPEN)
            ).alias("_session_anchor")
        )
        .with_columns(
            (
                (
                    (
                        pl.col("timestamp").dt.hour() * 60
                        + pl.col("timestamp").dt.minute()
                    )
                    - (SESSION_OPEN.hour * 60 + SESSION_OPEN.minute)
                )
                .floordiv(5)
                .clip(lower_bound=0)
            ).alias("_bucket_index")
        )
        .with_columns(
            (
                pl.col("timestamp").dt.date().cast(pl.String).str.to_date()
            ).alias("_date")
        )
        .group_by(["_date", "_bucket_index"], maintain_order=True)
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
                pl.col("_date").cast(pl.Datetime(time_zone=MARKET_TIMEZONE))
                + pl.duration(minutes=SESSION_OPEN.hour * 60 + SESSION_OPEN.minute)
                + pl.duration(minutes=5)
                * (pl.col("_bucket_index") + 1)
            ).alias("timestamp")
        )
        .with_columns(pl.col("_date").alias("session_date"))
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
    if df_5m.filter(
        (pl.col("high") < pl.col("low"))
        | (pl.col("open") < pl.col("low"))
        | (pl.col("open") > pl.col("high"))
        | (pl.col("close") < pl.col("low"))
        | (pl.col("close") > pl.col("high"))
    ).height:
        raise ValueError("Invalid 5m OHLC geometry")

    for session, expected in (
        ("session", len(expected_5m_close_timestamps)),
    ):
        _ = session, expected
