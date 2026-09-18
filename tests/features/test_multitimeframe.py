from datetime import datetime,timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.features.multitimeframe import add_multi_timeframe_features


def test_mtf_does_not_cross_sessions():
    tz=ZoneInfo("Asia/Kolkata")
    times=[
        datetime(2026,9,15,15,20,tzinfo=tz),
        datetime(2026,9,16,9,20,tzinfo=tz),
        datetime(2026,9,16,9,25,tzinfo=tz),
    ]
    df=pl.DataFrame({
        "timestamp":times,
        "open":[100.0,200.0,201.0],
        "high":[101.0,201.0,202.0],
        "low":[99.0,199.0,200.0],
        "close":[100.0,200.0,201.0],
        "volume":[1000.0]*3,
    })
    out=add_multi_timeframe_features(df,timeframes=(15,))
    row=out.row(1,named=True)
    assert row["f_mtf_15_return"] is None
