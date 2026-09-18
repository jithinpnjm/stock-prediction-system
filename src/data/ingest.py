from __future__ import annotations

from pathlib import Path

import polars as pl

from src.common.contracts import MARKET_TIMEZONE
from src.data.schemas import validate_schema


def _normalize_timestamp(series: pl.Series, input_timezone: str | None) -> pl.Series:
    if series.dtype == pl.String:
        series = series.str.to_datetime(strict=True)
    if not isinstance(series.dtype, pl.Datetime):
        raise TypeError(f"Unsupported timestamp dtype: {series.dtype}")
    if series.dtype.time_zone is None:
        if not input_timezone:
            raise ValueError("Naive timestamps require explicit input_timezone")
        series = series.dt.replace_time_zone(input_timezone)
    return series.dt.convert_time_zone(MARKET_TIMEZONE)


def normalize_ohlcv(df: pl.DataFrame, *, input_timezone: str | None = MARKET_TIMEZONE, timestamp_column: str = "timestamp") -> pl.DataFrame:
    if timestamp_column not in df.columns and "datetime" in df.columns:
        df = df.rename({"datetime": "timestamp"})
    if "timestamp" not in df.columns:
        raise ValueError("No timestamp/datetime column found")
    df = df.with_columns(_normalize_timestamp(df.get_column("timestamp"), input_timezone).alias("timestamp"))
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=True).alias(c) for c in ("open", "high", "low", "close", "volume")])
    df = df.with_columns(pl.col("timestamp").dt.date().alias("session_date"))
    df = df.select(["session_date", "timestamp", "open", "high", "low", "close", "volume"]).sort("timestamp")
    validate_schema(df, "1m")
    return df


def load_source(path: str | Path, *, input_timezone: str | None = MARKET_TIMEZONE) -> pl.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        df = pl.read_csv(p, try_parse_dates=True)
    elif p.suffix.lower() in {".parquet", ".pq"}:
        df = pl.read_parquet(p)
    else:
        raise ValueError(f"Unsupported source format: {p.suffix}")
    return normalize_ohlcv(df, input_timezone=input_timezone)


def write_bronze(df: pl.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p, compression="zstd")
