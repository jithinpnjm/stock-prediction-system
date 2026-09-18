from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from numba import njit


@njit
def _label_path(close, high, low, start_idx, end_idx, target_pts, stop_pts):
    n=len(start_idx)
    labels=np.zeros(n,dtype=np.int8)
    hit_idx=np.full(n,-1,dtype=np.int64)
    mfe_l=np.zeros(n); mae_l=np.zeros(n); mfe_s=np.zeros(n); mae_s=np.zeros(n)

    for i in range(n):
        entry=close[start_idx[i]-1]
        lt=entry+target_pts; ls=entry-stop_pts
        st=entry-target_pts; ss=entry+stop_pts
        lti=lsi=sti=ssi=-1
        ml=al=ms=ass=0.0

        for j in range(start_idx[i],end_idx[i]):
            up=high[j]-entry; down=entry-low[j]
            if up>ml: ml=up
            if down>al: al=down
            if down>ms: ms=down
            if up>ass: ass=up

            if lti==-1 and lsi==-1:
                both=high[j]>=lt and low[j]<=ls
                if both: lsi=j
                elif high[j]>=lt: lti=j
                elif low[j]<=ls: lsi=j
            if sti==-1 and ssi==-1:
                both=low[j]<=st and high[j]>=ss
                if both: ssi=j
                elif low[j]<=st: sti=j
                elif high[j]>=ss: ssi=j

        long_valid=lti!=-1 and (lsi==-1 or lti<lsi)
        short_valid=sti!=-1 and (ssi==-1 or sti<ssi)
        if long_valid and short_valid:
            if lti<=sti:
                labels[i]=1; hit_idx[i]=lti
            else:
                labels[i]=-1; hit_idx[i]=sti
        elif long_valid:
            labels[i]=1; hit_idx[i]=lti
        elif short_valid:
            labels[i]=-1; hit_idx[i]=sti

        mfe_l[i]=ml; mae_l[i]=al; mfe_s[i]=ms; mae_s[i]=ass
    return labels,hit_idx,mfe_l,mae_l,mfe_s,mae_s


def apply_triple_barrier_labels(
    events_5m:pl.DataFrame,
    bars_1m:pl.DataFrame,
    *,
    target_pts:float=200.0,
    stop_pts:float=70.0,
    max_horizon_minutes:int=375,
    direction:Literal["both","long","short"]="both",
)->pl.DataFrame:
    if events_5m.is_empty() or bars_1m.is_empty():
        raise ValueError("events_5m and bars_1m must be non-empty")
    events=events_5m.sort("timestamp")
    bars=bars_1m.sort("timestamp")
    e_ts=events["timestamp"].dt.epoch("ns").to_numpy()
    b_ts=bars["timestamp"].dt.epoch("ns").to_numpy()
    e_dates=events["timestamp"].dt.date().to_list()
    b_dates=bars["timestamp"].dt.date().to_list()
    starts=np.searchsorted(b_ts,e_ts,side="right").astype(np.int64)
    ends=np.empty(len(e_ts),dtype=np.int64)
    horizon_ns=np.int64(max_horizon_minutes)*60*1_000_000_000
    for i,ts in enumerate(e_ts):
        j=int(starts[i])
        end=int(np.searchsorted(b_ts,ts+horizon_ns,side="right"))
        while end>j and b_dates[end-1]!=e_dates[i]:
            end-=1
        ends[i]=max(j,end)
    if np.any(starts<=0) or np.any(ends<=starts):
        raise ValueError("Some events do not have a usable future source path")

    labels,hit_idx,mfe_l,mae_l,mfe_s,mae_s=_label_path(
        bars["close"].to_numpy(),bars["high"].to_numpy(),bars["low"].to_numpy(),
        starts,ends,target_pts,stop_pts
    )
    expiry_idx=np.maximum(ends-1,starts)
    barrier_idx=np.where(hit_idx>=0,hit_idx,expiry_idx)
    barrier_ns=b_ts[barrier_idx]
    expiry_ns=b_ts[expiry_idx]

    result=events.with_columns(
        pl.Series("label",labels,dtype=pl.Int8),
        pl.Series("event_end_timestamp",pl.from_numpy(expiry_ns).cast(pl.Datetime("ns",time_zone="Asia/Kolkata"))),
        pl.Series("barrier_timestamp",pl.from_numpy(barrier_ns).cast(pl.Datetime("ns",time_zone="Asia/Kolkata"))),
        pl.Series("mfe_long_points",mfe_l),
        pl.Series("mae_long_points",mae_l),
        pl.Series("mfe_short_points",mfe_s),
        pl.Series("mae_short_points",mae_s),
        pl.lit(target_pts).alias("target_points"),
        pl.lit(stop_pts).alias("stop_points"),
        pl.lit(max_horizon_minutes).alias("max_horizon_minutes"),
    )
    if direction=="long":
        result=result.with_columns(pl.when(pl.col("label")==1).then(1).otherwise(0).cast(pl.Int8).alias("label"))
    elif direction=="short":
        result=result.with_columns(pl.when(pl.col("label")==-1).then(-1).otherwise(0).cast(pl.Int8).alias("label"))
    from .mfe_mae import add_excursion_features
    return add_excursion_features(result)
