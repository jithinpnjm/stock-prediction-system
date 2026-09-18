from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.features.microstructure import add_1m_inside_5m_features


def test_1m_path_summary_is_causal_to_5m_close():
    tz = ZoneInfo("Asia/Kolkata")
    base = datetime(2026, 9, 15, 9, 16, tzinfo=tz)
    one_times = [base + timedelta(minutes=i) for i in range(5)]
    one = pl.DataFrame(
        {
            "timestamp": one_times,
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 106],
            "low": [99, 100, 101, 102, 103],
            "close": [101, 102, 103, 104, 105],
            "volume": [1, 2, 3, 4, 5],
        }
    )
    five = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 9, 15, 9, 20, tzinfo=tz)],
            "open": [100.0],
            "high": [106.0],
            "low": [99.0],
            "close": [105.0],
            "volume": [15.0],
        }
    )
    out = add_1m_inside_5m_features(one, five)
    row = out.row(0, named=True)
    assert row["f_1m_count"] == 5
    # path high=106, path low=99, last close=105
    assert row["f_1m_path_range"] == 7
    assert row["f_1m_path_high_rejection"] == 1
