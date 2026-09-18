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
