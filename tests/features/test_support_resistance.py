from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.features.support_resistance import add_support_resistance_features


def _session(n: int = 75):
    tz = ZoneInfo("Asia/Kolkata")
    times = [datetime(2026, 9, 15, 9, 20, tzinfo=tz) + timedelta(minutes=5 * i) for i in range(n)]
    close = [100.0 + (i % 5) for i in range(n)]
    return pl.DataFrame(
        {
            "timestamp": times,
            "open": close,
            "high": [c + 1 for c in close],
            "low": [c - 1 for c in close],
            "close": close,
            "volume": [1000.0] * n,
        }
    )


def test_touch_counts_are_populated_once_warmed_up():
    """A lookback window that spans most of a session must still produce
    non-null touch counts once resistance/support have warmed up,
    rather than never having a full non-null window within the day."""
    out = add_support_resistance_features(_session(75), lookback=48)
    warmed_up = out.tail(10)
    assert warmed_up["f_resistance_touches"].null_count() == 0
    assert warmed_up["f_support_touches"].null_count() == 0
