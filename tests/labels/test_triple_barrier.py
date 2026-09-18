from datetime import datetime,timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.labels.triple_barrier import apply_triple_barrier_labels


def test_long_target_path():
    tz=ZoneInfo("Asia/Kolkata")
    times=[datetime(2026,9,15,9,20,tzinfo=tz)+timedelta(minutes=i) for i in range(12)]
    close=[100]*12; high=[101]*12; low=[99]*12
    high[1]=301
    bars=pl.DataFrame({"timestamp":times,"open":close,"high":high,"low":low,"close":close,"volume":[1000]*12})
    events=bars.filter(pl.col("timestamp")==times[0])
    out=apply_triple_barrier_labels(events,bars,target_pts=200,stop_pts=70,max_horizon_minutes=30)
    assert out["label"][0]==1
    assert out["mfe_points"][0]>=200


def test_short_target_path():
    tz=ZoneInfo("Asia/Kolkata")
    times=[datetime(2026,9,15,9,20,tzinfo=tz)+timedelta(minutes=i) for i in range(12)]
    close=[500]*12; high=[501]*12; low=[499]*12
    low[1]=299
    bars=pl.DataFrame({"timestamp":times,"open":close,"high":high,"low":low,"close":close,"volume":[1000]*12})
    events=bars.filter(pl.col("timestamp")==times[0])
    out=apply_triple_barrier_labels(events,bars,target_pts=200,stop_pts=70,max_horizon_minutes=30)
    assert out["label"][0]==-1
    assert out["mfe_points"][0]>=200
