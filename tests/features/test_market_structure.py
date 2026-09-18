from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.features.candles import add_candle_geometry_features
from src.features.market_structure import add_market_structure_features


def _bars(first_day_values):
    tz = ZoneInfo("Asia/Kolkata")
    d1 = datetime(2026, 9, 15, 9, 20, tzinfo=tz)
    d2 = datetime(2026, 9, 16, 9, 20, tzinfo=tz)
    times = [
        d1 + timedelta(minutes=5 * i) for i in range(len(first_day_values))
    ] + [
        d2 + timedelta(minutes=5 * i) for i in range(len(first_day_values))
    ]
    close = first_day_values + [x + 100 for x in first_day_values]
    return pl.DataFrame(
        {
            "timestamp": times,
            "open": close,
            "high": [x + 1 for x in close],
            "low": [x - 1 for x in close],
            "close": close,
            "volume": [1000.0] * len(close),
        }
    )


def test_market_structure_is_causal():
    df = _bars([100, 102, 101, 104])
    out = add_market_structure_features(add_candle_geometry_features(df))
    assert out.height == 8
    assert out["f_structure_trend"][0] == 0


def test_market_structure_resets_at_session_boundary():
    df = _bars([100, 102, 101])
    out = add_market_structure_features(add_candle_geometry_features(df))
    second_day = out.filter(
        pl.col("timestamp").dt.date() == datetime(2026, 9, 16).date()
    )
    assert second_day["f_last_swing_high"][0] is None
    assert second_day["f_last_swing_low"][0] is None
    assert second_day["f_structure_trend"][0] == 0
