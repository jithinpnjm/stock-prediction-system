from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.data.calendar import NSECalendar
from src.data.validate import validate_ohlcv


def _df(rows: int = 6) -> pl.DataFrame:
    tz = ZoneInfo("Asia/Kolkata")
    start = datetime(2026, 1, 5, 9, 16, tzinfo=tz)
    ts = [start + timedelta(minutes=i) for i in range(rows)]
    return pl.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [100.5] * rows,
            "volume": [10.0] * rows,
        }
    )


def _gates(df: pl.DataFrame):
    return {
        report["gate"]: report
        for report in validate_ohlcv(df, calendar=NSECalendar())
    }


def test_validation_can_report_incomplete_session():
    gates = _gates(_df())
    assert gates["non_empty"]["passed"]
    assert gates["session_completeness"]["passed"] is False


def test_validation_rejects_duplicate_timestamp():
    df = pl.concat([_df(), _df().head(1)])
    gates = _gates(df)
    assert gates["no_duplicate_timestamps"]["count"] == 1
    assert gates["no_duplicate_timestamps"]["passed"] is False


def test_validation_rejects_bad_geometry():
    df = _df().with_columns(pl.lit(102.0).alias("low"))
    gates = _gates(df)
    assert gates["ohlcv_geometry"]["count"] > 0
    assert gates["ohlcv_geometry"]["passed"] is False
