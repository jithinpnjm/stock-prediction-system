from datetime import datetime,timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.data.aggregate import aggregate_1m_to_5m,validate_aggregation
from src.data.calendar import NSECalendar


def test_5m_bucket_close_semantics():
    tz=ZoneInfo("Asia/Kolkata")
    times=[datetime(2026,9,15,9,16,tzinfo=tz)+timedelta(minutes=i) for i in range(10)]
    prices=list(range(100,110))
    df=pl.DataFrame({
        "timestamp":times,
        "open":prices,"high":[p+1 for p in prices],
        "low":[p-1 for p in prices],"close":prices,"volume":[100]*10
    })
    out=aggregate_1m_to_5m(df,calendar=NSECalendar())
    validate_aggregation(out)
    assert out["timestamp"].to_list()==[
        datetime(2026,9,15,9,20,tzinfo=tz),
        datetime(2026,9,15,9,25,tzinfo=tz),
    ]
    assert out["open"].to_list()==[100.0,105.0]


def test_incomplete_bucket_is_dropped():
    tz=ZoneInfo("Asia/Kolkata")
    times=[datetime(2026,9,15,9,16,tzinfo=tz)+timedelta(minutes=i) for i in range(4)]
    df=pl.DataFrame({
        "timestamp":times,"open":[100.0]*4,"high":[101.0]*4,
        "low":[99.0]*4,"close":[100.0]*4,"volume":[100.0]*4
    })
    out=aggregate_1m_to_5m(df,calendar=NSECalendar())
    assert out.is_empty()
