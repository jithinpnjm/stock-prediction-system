from datetime import datetime,timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from src.data.calendar import NSECalendar
from src.data.validate import assert_valid


def _day():
    tz=ZoneInfo("Asia/Kolkata")
    start=datetime(2026,9,15,9,16,tzinfo=tz)
    ts=[start+timedelta(minutes=i) for i in range(375)]
    return pl.DataFrame({
        "timestamp":ts,
        "open":[100.0]*375,"high":[101.0]*375,"low":[99.0]*375,
        "close":[100.0]*375,"volume":[1000.0]*375,
    })


def test_full_session_passes():
    df=_day()
    cal=NSECalendar()
    reports=assert_valid(df,calendar=cal)
    assert all(r["passed"] for r in reports)


def test_duplicate_fails():
    df=_day()
    df=pl.concat([df,df.head(1)])
    with pytest.raises(ValueError,match="duplicate"):
        assert_valid(df.sort("timestamp"),calendar=NSECalendar())
