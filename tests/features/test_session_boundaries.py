from datetime import datetime,timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.features.candles import add_candle_geometry_features
from src.features.clusters import add_candle_cluster_features


def _bars():
    tz=ZoneInfo("Asia/Kolkata")
    d1=datetime(2026,9,15,9,20,tzinfo=tz)
    d2=datetime(2026,9,16,9,20,tzinfo=tz)
    times=[d1+timedelta(minutes=5*i) for i in range(3)]+[d2+timedelta(minutes=5*i) for i in range(3)]
    close=[100,101,102,200,201,202]
    return pl.DataFrame({
        "timestamp":times,"open":close,"high":[x+1 for x in close],
        "low":[x-1 for x in close],"close":close,"volume":[1000]*6
    })


def test_cluster_does_not_cross_sessions():
    df=add_candle_cluster_features(add_candle_geometry_features(_bars()),max_bars=3)
    second_day_first=df.filter(pl.col("timestamp")==datetime(2026,9,16,9,20,tzinfo=ZoneInfo("Asia/Kolkata"))).row(0,named=True)
    assert second_day_first["f_cluster_2_return"] is None
