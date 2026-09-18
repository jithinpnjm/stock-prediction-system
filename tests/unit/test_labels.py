from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl

from src.common.contracts import MARKET_TIMEZONE, LabelConfig
from src.labels.triple_barrier import apply_triple_barrier_labels


def _event():
    tz = ZoneInfo(MARKET_TIMEZONE)
    return pl.DataFrame({"session_date": [datetime(2026, 1, 5, 9, 20, tzinfo=tz).date()], "timestamp": [datetime(2026, 1, 5, 9, 20, tzinfo=tz)], "open": [1000.0], "high": [1000.0], "low": [1000.0], "close": [1000.0], "volume": [1.0]})


def test_long_target_uses_1m_path_and_entry_open():
    tz = ZoneInfo(MARKET_TIMEZONE)
    one = pl.DataFrame({"timestamp": [datetime(2026, 1, 5, 9, 21, tzinfo=tz), datetime(2026, 1, 5, 9, 22, tzinfo=tz)], "open": [1005.0, 1005.0], "high": [1190.0, 1210.0], "low": [995.0, 995.0], "close": [1005.0, 1205.0], "volume": [1.0, 1.0]})
    out = apply_triple_barrier_labels(_event(), one, LabelConfig(target_points=200, stop_points=70, horizon_bars=2))
    assert out["entry_price"][0] == 1005.0
    assert out["label"][0] == 1


def test_intrabar_collision_is_ambiguous():
    tz = ZoneInfo(MARKET_TIMEZONE)
    one = pl.DataFrame({"timestamp": [datetime(2026, 1, 5, 9, 21, tzinfo=tz)], "open": [1000.0], "high": [1210.0], "low": [920.0], "close": [1000.0], "volume": [1.0]})
    out = apply_triple_barrier_labels(_event(), one, LabelConfig(target_points=200, stop_points=70, horizon_bars=2))
    assert out["label"][0] == 0
    assert out["barrier_type"][0] == "ambiguous"
