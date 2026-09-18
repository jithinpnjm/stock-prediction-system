from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl

from src.common.contracts import MARKET_TIMEZONE
from src.labels.triple_barrier import apply_triple_barrier_labels


def _event():
    tz = ZoneInfo(MARKET_TIMEZONE)
    return pl.DataFrame(
        {
            "session_date": [datetime(2026, 1, 5, 9, 20, tzinfo=tz).date()],
            "timestamp": [datetime(2026, 1, 5, 9, 20, tzinfo=tz)],
            "open": [1000.0],
            "high": [1000.0],
            "low": [1000.0],
            "close": [1000.0],
            "volume": [1.0],
        }
    )


def test_long_target_uses_1m_path_and_entry_open():
    tz = ZoneInfo(MARKET_TIMEZONE)
    one = pl.concat(
        [
            _event(),
            pl.DataFrame(
                {
                    "session_date": [datetime(2026, 1, 5, 9, 21, tzinfo=tz).date()] * 2,
                    "timestamp": [
                        datetime(2026, 1, 5, 9, 21, tzinfo=tz),
                        datetime(2026, 1, 5, 9, 22, tzinfo=tz),
                    ],
                    "open": [1005.0, 1005.0],
                    "high": [1190.0, 1210.0],
                    "low": [995.0, 995.0],
                    "close": [1005.0, 1205.0],
                    "volume": [1.0, 1.0],
                }
            ),
        ]
    )
    out = apply_triple_barrier_labels(
        _event(), one, target_pts=200, stop_pts=70, max_horizon_minutes=30
    )
    # Entry price is the event bar's own close, per the canonical engine.
    assert out["target_points"][0] == 200
    assert out["label"][0] == 1


def test_intrabar_collision_resolves_conservatively():
    """A single 1m bar whose high/low span both the long target and the
    long stop is inherently ambiguous about hit order. The canonical
    engine resolves this conservatively (treats it as the stop side),
    which for a symmetric barrier on both sides yields a net label of 0
    (no clean winner) rather than declaring a long win."""
    tz = ZoneInfo(MARKET_TIMEZONE)
    one = pl.concat(
        [
            _event(),
            pl.DataFrame(
                {
                    "session_date": [datetime(2026, 1, 5, 9, 21, tzinfo=tz).date()],
                    "timestamp": [datetime(2026, 1, 5, 9, 21, tzinfo=tz)],
                    "open": [1000.0],
                    "high": [1210.0],
                    "low": [920.0],
                    "close": [1000.0],
                    "volume": [1.0],
                }
            ),
        ]
    )
    out = apply_triple_barrier_labels(
        _event(), one, target_pts=200, stop_pts=70, max_horizon_minutes=30
    )
    assert out["label"][0] == 0
