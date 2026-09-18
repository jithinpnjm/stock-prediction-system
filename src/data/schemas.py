from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.common.contracts import (
    EXPECTED_1M_BARS,
    EXPECTED_5M_BARS,
    MARKET_TIMEZONE,
    SESSION_CLOSE,
    SESSION_OPEN,
)


CANONICAL_1M_SCHEMA = {
    "session_date": pl.Date,
    "timestamp": pl.Datetime(time_zone=MARKET_TIMEZONE),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}

CANONICAL_5M_SCHEMA = {
    "session_date": pl.Date,
    "timestamp": pl.Datetime(time_zone=MARKET_TIMEZONE),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


@dataclass(frozen=True)
class MarketSession:
    session_date: object
    open_time: object
    close_time: object


def required_ohlcv_columns() -> tuple[str, ...]:
    return ("timestamp", "open", "high", "low", "close", "volume")


def expected_bars(timeframe: str) -> int:
    if timeframe == "1m":
        return EXPECTED_1M_BARS
    if timeframe == "5m":
        return EXPECTED_5M_BARS
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def validate_schema(df: pl.DataFrame, timeframe: str) -> None:
    required = set(required_ohlcv_columns())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.is_empty():
        raise ValueError("Dataset is empty")

    if df.schema["timestamp"] != pl.Datetime(time_zone=MARKET_TIMEZONE):
        raise TypeError(
            f"timestamp must be timezone-aware {MARKET_TIMEZONE}; "
            f"got {df.schema['timestamp']}"
        )

    for column in ("open", "high", "low", "close", "volume"):
        if not df.schema[column].is_numeric():
            raise TypeError(f"{column} must be numeric")


def in_session_expr() -> pl.Expr:
    return (
        (pl.col("timestamp").dt.time() >= SESSION_OPEN)
        & (pl.col("timestamp").dt.time() < SESSION_CLOSE)
    )
