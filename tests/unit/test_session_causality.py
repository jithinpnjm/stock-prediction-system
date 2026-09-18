from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.features.clusters import add_consecutive_cluster_features
from src.features.event_sampling import add_cusum_events
from src.features.volatility import add_volatility_features


def _two_sessions() -> pl.DataFrame:
    tz = ZoneInfo("Asia/Kolkata")
    first = datetime(2026, 1, 5, 9, 15, tzinfo=tz)
    second = datetime(2026, 1, 6, 9, 15, tzinfo=tz)
    timestamps = [
        first,
        first + timedelta(minutes=5),
        second,
        second + timedelta(minutes=5),
    ]
    return pl.DataFrame(
        {
            "session_date": [
                first.date(),
                first.date(),
                second.date(),
                second.date(),
            ],
            "timestamp": timestamps,
            "open": [100.0, 101.0, 200.0, 201.0],
            "high": [101.0, 102.0, 201.0, 202.0],
            "low": [99.0, 100.0, 199.0, 200.0],
            "close": [101.0, 102.0, 201.0, 202.0],
            "volume": [10.0, 10.0, 20.0, 20.0],
        }
    )


def test_clusters_reset_at_session_boundary():
    out = add_consecutive_cluster_features(_two_sessions())
    assert out["consecutive_length"].to_list()[2] == 1


def test_volatility_resets_at_session_boundary():
    out = add_volatility_features(_two_sessions(), periods=(2,))
    assert out["f_atr_2"][2] is None
    assert out["f_abs_return_2"][2] is None


def test_cusum_resets_at_session_boundary():
    df = _two_sessions().with_columns(pl.Series("atr_14", [1.0, 1.0, 1.0, 1.0]))
    out = add_cusum_events(df, threshold_atr=100.0)
    assert out["cusum_event"].to_list()[2] == 0
