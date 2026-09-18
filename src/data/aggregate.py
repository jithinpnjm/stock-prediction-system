from __future__ import annotations

from datetime import time

import polars as pl

from src.common.contracts import (
    MARKET_TIMEZONE,
    SESSION_OPEN,
)
from src.data.calendar import SessionSpec


def aggregate_1m_to_5m(
    df_1m: pl.DataFrame,
    *,
    session_overrides: dict | None = None,
    drop_incomplete: bool = True,
) -> pl.DataFrame:
    if df_1m.is_empty():
        raise ValueError(
            "Cannot aggregate empty 1m data"
        )

    overrides = session_overrides or {}

    # Build a small session calendar mapping. Standard sessions use 09:15.
    calendar_rows = [
        {
            "session_date": session,
            "_open_minutes": (
                spec.session_open.hour * 60
                + spec.session_open.minute
            ),
            "_expected_5m_bars": max(
                1,
                spec.expected_1m_bars // 5,
            ),
        }
        for session, spec in overrides.items()
    ]
    calendar = (
        pl.DataFrame(calendar_rows)
        if calendar_rows
        else pl.DataFrame(
            schema={
                "session_date": pl.Date,
                "_open_minutes": pl.Int64,
                "_expected_5m_bars": pl.Int64,
            }
        )
    )

    out = (
        df_1m.sort("timestamp")
        .with_columns(
            pl.col("timestamp")
            .dt.date()
            .alias("session_date")
        )
        .join(
            calendar,
            on="session_date",
            how="left",
        )
        .with_columns(
            pl.col("_open_minutes").fill_null(
                SESSION_OPEN.hour * 60
                + SESSION_OPEN.minute
            ),
            pl.col("_expected_5m_bars").fill_null(75),
        )
        .with_columns(
            (
                (
                    pl.col("timestamp").dt.hour() * 60
                    + pl.col("timestamp").dt.minute()
                    - pl.col("_open_minutes")
                )
                / 5.0
            )
            .floor()
            .cast(pl.Int64)
            .alias("_bucket_index")
        )
        .filter(
            pl.col("_bucket_index").is_between(
                0,
                pl.col("_expected_5m_bars") - 1,
            )
        )
        .group_by(
            ["session_date", "_open_minutes", "_bucket_index"],
            maintain_order=True,
        )
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
                    pl.Datetime(
                        time_zone=MARKET_TIMEZONE
                    )
                )
                + pl.duration(
                    minutes=pl.col("_open_minutes")
                )
                + pl.duration(
                    minutes=5
                ) * (
                    pl.col("_bucket_index") + 1
                )
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
        out = out.filter(
            pl.col("source_1m_count") == 5
        )

    if out.is_empty():
        raise ValueError(
            "No complete 5m buckets after aggregation"
        )
    return out


def validate_aggregation(
    df_5m: pl.DataFrame,
    *,
    session_overrides: dict | None = None,
) -> None:
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
        raise ValueError(
            f"Invalid 5m OHLC rows: {invalid.height}"
        )

    overrides = session_overrides or {}
    expected_counts = []
    for session in (
        df_5m.get_column("session_date")
        .unique()
        .to_list()
    ):
        spec = overrides.get(
            session,
            SessionSpec(
                session_open=SESSION_OPEN,
                session_close=time(15, 30),
                expected_1m_bars=375,
            ),
        )
        expected_counts.append(
            {
                "session_date": session,
                "expected_5m": max(
                    1,
                    spec.expected_1m_bars // 5,
                ),
            }
        )

    expected_frame = pl.DataFrame(
        expected_counts
    )
    counts = df_5m.group_by(
        "session_date"
    ).agg(
        pl.len().alias("actual_5m")
    )
    bad = counts.join(
        expected_frame,
        on="session_date",
    ).filter(
        pl.col("actual_5m")
        != pl.col("expected_5m")
    )
    if bad.height:
        raise ValueError(
            f"Unexpected 5m session counts: {bad.to_dicts()}"
        )
