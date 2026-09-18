from __future__ import annotations

import numpy as np
import polars as pl


def cusum_events(
    returns: np.ndarray,
    thresholds: np.ndarray | float,
) -> np.ndarray:
    r=np.asarray(returns,dtype=float)
    t=np.full(len(r),float(thresholds)) if np.isscalar(thresholds) else np.asarray(thresholds,dtype=float)
    pos=neg=0.0
    events=np.zeros(len(r),dtype=np.int8)
    for i,x in enumerate(np.nan_to_num(r,nan=0.0)):
        threshold=max(float(t[i]),1e-9)
        pos=max(0.0,pos+x); neg=min(0.0,neg+x)
        if pos>threshold:
            events[i]=1; pos=0.0; neg=0.0
        elif neg<-threshold:
            events[i]=1; pos=0.0; neg=0.0
    return events


def add_event_sampling_features(
    df:pl.DataFrame,
    threshold_multiple:float=2.0,
    volatility_lookback:int=60,
)->pl.DataFrame:
    out=df.sort("timestamp")
    if "f_return_1" not in out.columns:
        out=out.with_columns(
            (pl.col("close")/pl.col("close").shift(1)-1).alias("f_return_1")
        )
    ret=out["f_return_1"].to_numpy()
    scale=(
        out["f_return_1"].rolling_std(volatility_lookback).shift(1).to_numpy()
    )
    threshold=np.nan_to_num(scale,nan=np.nanmedian(scale) if np.isfinite(scale).any() else 1e-4)
    events=cusum_events(ret,threshold*threshold_multiple)
    return out.with_columns(
        pl.Series("f_cusum_event",events,dtype=pl.Int8),
        pl.Series("f_cusum_threshold",threshold*threshold_multiple),
    )
