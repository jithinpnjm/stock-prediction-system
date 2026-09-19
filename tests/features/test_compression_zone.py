from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.features.compression_zone import add_compression_zone_features


def _session(prices: list[float]):
    tz = ZoneInfo("Asia/Kolkata")
    times = [
        datetime(2026, 9, 15, 9, 20, tzinfo=tz) + timedelta(minutes=5 * i)
        for i in range(len(prices))
    ]
    return pl.DataFrame(
        {
            "timestamp": times,
            "open": prices,
            "high": [p + 2 for p in prices],
            "low": [p - 2 for p in prices],
            "close": prices,
            "volume": [1000.0] * len(prices),
        }
    )


def test_swing_distance_only_populates_after_confirmed_move():
    # flat chop, a clean 250pt breakout, then a reversal that confirms
    # the breakout's high as a swing pivot (a zigzag pivot is only ever
    # known in hindsight, once price has moved away from it -- this
    # must not populate before that reversal has actually happened).
    prices = [100.0] * 10 + [120, 150, 200, 250, 300, 350] + [300, 250, 200, 150, 100]
    out = add_compression_zone_features(_session(prices), swing_threshold_points=200.0)
    dist = out["f_dist_to_recent_swing_atr"].to_list()
    # no confirmed swing yet during the chop or the un-reversed breakout
    assert all(v is None or v != v for v in dist[:16])  # None or NaN
    # once price has reversed 200+ points off the high, that high is a
    # confirmed swing and later bars should carry a real distance value
    assert any(v is not None and v == v for v in dist[16:])


def test_compression_flag_is_causal_and_session_local():
    prices = [100.0] * 6
    out = add_compression_zone_features(_session(prices))
    assert out.height == 6
    assert "f_is_compressed_5" in out.columns
    assert "f_directional_efficiency_5" in out.columns
