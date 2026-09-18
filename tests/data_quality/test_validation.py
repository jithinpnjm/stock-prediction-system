from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.common.contracts import MARKET_TIMEZONE
from src.data.calendar import NSECalendar
from src.data.validate import validate_ohlcv


def _df(rows: int = 6):
    tz = ZoneInfo(MARKET_TIMEZONE)
    start = datetime(2026, 1, 5, 9, 16, tzinfo=tz)
    ts = [start + timedelta(minutes=i) for i in range(rows)]
    return pl.DataFrame(
        {
            "session_date": [date(2026, 1, 5)] * rows,
            "timestamp": ts,
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [100.5] * rows,
            "volume": [10.0] * rows,
        }
    )


def _gate(reports, name):
    return next(r for r in reports if r["gate"] == name)


def test_validation_can_report_incomplete_session():
    df = _df()
    reports = validate_ohlcv(df, calendar=NSECalendar())
    assert df.height == 6
    assert _gate(reports, "no_duplicate_timestamps")["passed"]
    assert _gate(reports, "ohlcv_geometry")["passed"]
    # A 6-row slice of a session is not a complete trading day.
    assert not _gate(reports, "session_completeness")["passed"]


def test_validation_rejects_duplicate_timestamp():
    df = pl.concat([_df(), _df().head(1)])
    reports = validate_ohlcv(df, calendar=NSECalendar())
    gate = _gate(reports, "no_duplicate_timestamps")
    assert gate["count"] == 1
    assert not gate["passed"]


def test_validation_rejects_bad_geometry():
    df = _df().with_columns(pl.lit(102.0).alias("low"))
    reports = validate_ohlcv(df, calendar=NSECalendar())
    gate = _gate(reports, "ohlcv_geometry")
    assert gate["count"] > 0
    assert not gate["passed"]
