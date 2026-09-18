from datetime import datetime,timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.features.candles import add_candle_geometry_features
from src.features.market_structure import add_market_structure_features


def test_market_structure_is_causal():
    tz=ZoneInfo("Asia/Kolkata")
    times=[datetime(2026,9,15,9,20,tzinfo=tz)+timedelta(minutes=5*i) for i in range(8)]
    close=[100,102,101,104,103,106,105,108]
    df=pl.DataFrame({
        "timestamp":times,"open":close,"high":[x+1 for x in close],
        "low":[x-1 for x in close],"close":close,"volume":[1000]*8
    })
    out=add_market_structure_features(add_candle_geometry_features(df))
    assert out.height==8
    assert out["f_structure_trend"][0]==0
