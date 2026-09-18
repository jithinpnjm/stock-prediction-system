from datetime import datetime,timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.features.candles import add_candle_geometry_features
from src.features.clusters import add_candle_cluster_features


def _bars(n=20):
    tz=ZoneInfo("Asia/Kolkata")
    ts=[datetime(2026,9,15,9,20,tzinfo=tz)+timedelta(minutes=5*i) for i in range(n)]
    close=[100+i for i in range(n)]
    return pl.DataFrame({
        "timestamp":ts,
        "open":[x-0.5 for x in close],
        "high":[x+1 for x in close],
        "low":[x-1 for x in close],
        "close":close,
        "volume":[1000]*n,
    })


def test_causal_features_do_not_change_past_when_future_is_appended():
    full=add_candle_cluster_features(add_candle_geometry_features(_bars(20)),max_bars=10)
    past=add_candle_cluster_features(add_candle_geometry_features(_bars(10)),max_bars=10)
    full_last=full.row(9,named=True)
    past_last=past.row(9,named=True)
    for name in ["f_body","f_range","f_cluster_3_return","f_cluster_10_return"]:
        assert full_last[name]==past_last[name]

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.features.event_sampling import add_event_sampling_features


def test_cusum_threshold_does_not_use_future_session_volatility():
    tz = ZoneInfo("Asia/Kolkata")
    start = datetime(2026, 9, 15, 9, 20, tzinfo=tz)
    base = pl.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=5 * i) for i in range(6)],
            "open": [100.0, 100.1, 100.2, 100.3, 100.4, 100.5],
            "high": [100.2, 100.3, 100.4, 100.5, 100.6, 100.7],
            "low": [99.8, 99.9, 100.0, 100.1, 100.2, 100.3],
            "close": [100.1, 100.2, 100.3, 100.4, 100.5, 100.6],
            "volume": [100.0] * 6,
        }
    )
    future = base.with_columns(
        pl.when(pl.col("timestamp") == base["timestamp"][-1])
        .then(pl.col("high") + 1000.0)
        .otherwise(pl.col("high"))
        .alias("high")
    )

    a = add_event_sampling_features(
        base,
        threshold_multiple=2.0,
        volatility_lookback=3,
    )
    b = add_event_sampling_features(
        future,
        threshold_multiple=2.0,
        volatility_lookback=3,
    )

    assert a["f_cusum_threshold"][:4].to_list() == (
        b["f_cusum_threshold"][:4].to_list()
    )
