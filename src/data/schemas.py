from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")

CANONICAL_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime(time_zone="Asia/Kolkata"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


@dataclass(frozen=True)
class DataContract:
    symbol: str = "NSE:NIFTYBANK-INDEX"
    source_timeframe: str = "1m"
    model_timeframe: str = "5m"
    timezone: str = "Asia/Kolkata"
    session_open: str = "09:15"
    session_close: str = "15:30"
    evaluation_open: str = "09:30"
    evaluation_close: str = "15:00"


def normalize_timestamp(expr: pl.Expr) -> pl.Expr:
    """Return a timezone-aware Asia/Kolkata timestamp expression."""
    return (
        expr.cast(pl.Datetime(time_zone=None))
        .dt.replace_time_zone("Asia/Kolkata")
    )


def validate_required_schema(df: pl.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    expected_numeric = ("open", "high", "low", "close", "volume")
    for column in expected_numeric:
        if not df[column].dtype.is_numeric():
            raise TypeError(f"{column} must be numeric, got {df[column].dtype}")


def coerce_canonical_schema(df: pl.DataFrame) -> pl.DataFrame:
    validate_required_schema(df)
    return (
        df.with_columns(
            normalize_timestamp(pl.col("timestamp")).alias("timestamp"),
            *[
                pl.col(c).cast(pl.Float64, strict=False).alias(c)
                for c in ("open", "high", "low", "close", "volume")
            ],
        )
        .select(list(REQUIRED_COLUMNS))
        .sort("timestamp")
    )
