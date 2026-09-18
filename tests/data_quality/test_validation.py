from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from src.common.contracts import MARKET_TIMEZONE
from src.data.validate import validate_1m


def _df(rows: int = 6):
    tz = ZoneInfo(MARKET_TIMEZONE)
    start = datetime(2026, 1, 5, 9, 15, tzinfo=tz)
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


def test_validation_can_report_incomplete_session():
    report = validate_1m(_df(), require_complete_sessions=False)
    assert report.rows == 6
    assert report.ok


def test_validation_rejects_duplicate_timestamp():
    df = _df()
    df = pl.concat([df, df.head(1)])
    report = validate_1m(df, require_complete_sessions=False)
    assert report.duplicate_timestamps == 1
    assert not report.ok


def test_validation_rejects_bad_geometry():
    df = _df().with_columns(pl.lit(102.0).alias("low"))
    report = validate_1m(df, require_complete_sessions=False)
    assert report.invalid_geometry > 0
    assert not report.ok
