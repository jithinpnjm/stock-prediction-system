from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.common.contracts import MARKET_TIMEZONE
from src.data.aggregate import aggregate_1m_to_5m
from src.data.calendar import NSECalendar


def test_5m_candle_uses_close_timestamp():
    tz = ZoneInfo(MARKET_TIMEZONE)
    start = datetime(2026, 1, 5, 9, 16, tzinfo=tz)
    ts = [start + timedelta(minutes=i) for i in range(10)]
    df = pl.DataFrame(
        {
            "timestamp": ts,
            "open": [100 + i for i in range(10)],
            "high": [101 + i for i in range(10)],
            "low": [99 + i for i in range(10)],
            "close": [100.5 + i for i in range(10)],
            "volume": [1.0] * 10,
        }
    )
    out = aggregate_1m_to_5m(
        df,
        calendar=NSECalendar(),
        drop_incomplete=True,
    )
    assert out.height == 2
    assert out["timestamp"][0] == datetime(2026, 1, 5, 9, 20, tzinfo=tz)
    assert out["timestamp"][1] == datetime(2026, 1, 5, 9, 25, tzinfo=tz)
    assert out["source_1m_count"].to_list() == [5, 5]
