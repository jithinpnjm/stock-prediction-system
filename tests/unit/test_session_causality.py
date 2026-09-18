from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.features.clusters import add_candle_cluster_features
from src.features.candles import add_candle_geometry_features
from src.features.event_sampling import add_event_sampling_features
from src.features.volatility import add_volatility_features


def _two_sessions() -> pl.DataFrame:
    tz = ZoneInfo("Asia/Kolkata")
    first = datetime(2026, 1, 5, 9, 20, tzinfo=tz)
    second = datetime(2026, 1, 6, 9, 20, tzinfo=tz)
    timestamps = [
        first,
        first + timedelta(minutes=5),
        second,
        second + timedelta(minutes=5),
    ]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 101.0, 200.0, 201.0],
            "high": [101.0, 102.0, 201.0, 202.0],
            "low": [99.0, 100.0, 199.0, 200.0],
            "close": [101.0, 102.0, 201.0, 202.0],
            "volume": [10.0, 10.0, 20.0, 20.0],
        }
    )


def test_clusters_reset_at_session_boundary():
    out = add_candle_cluster_features(
        add_candle_geometry_features(_two_sessions()),
        max_bars=2,
    )
    assert out["f_cluster_2_return"][2] is None


def test_volatility_resets_at_session_boundary():
    out = add_volatility_features(_two_sessions(), periods=(2, 14))
    assert out["f_atr_2"][2] is None
    assert out["f_abs_return_2"][2] is None


def test_cusum_resets_at_session_boundary():
    out = add_event_sampling_features(
        add_candle_geometry_features(_two_sessions()),
        threshold_multiple=100.0,
        volatility_lookback=2,
    )
    assert out["f_return_1"][2] is None
    assert out["f_cusum_event"][2] == 0
