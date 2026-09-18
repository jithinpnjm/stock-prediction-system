from datetime import datetime,timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.features.historical_intraday import add_historical_intraday_context


def test_historical_slot_uses_only_prior_days():
    tz=ZoneInfo("Asia/Kolkata")
    rows=[]
    for day in range(3):
        d=15+day
        rows.append({
            "timestamp":datetime(2026,9,d,9,20,tzinfo=tz),
            "open":100+day,"high":101+day,"low":99+day,
            "close":100+day,"volume":1000,
            "f_session_bar_index":0,
        })
    df=pl.DataFrame(rows)
    out=add_historical_intraday_context(df,lookback_days=2)
    assert out["f_historical_slot_return"][0] is None
