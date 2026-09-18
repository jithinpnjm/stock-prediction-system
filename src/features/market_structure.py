from __future__ import annotations

import numpy as np
import polars as pl


def _state_arrays(
    high:np.ndarray,
    low:np.ndarray,
    swing_high:np.ndarray,
    swing_low:np.ndarray,
):
    n=len(high)
    last_high=np.full(n,np.nan)
    last_low=np.full(n,np.nan)
    hh=np.zeros(n,dtype=np.int8); lh=np.zeros(n,dtype=np.int8)
    hl=np.zeros(n,dtype=np.int8); ll=np.zeros(n,dtype=np.int8)
    trend=np.zeros(n,dtype=np.int8)
    prev_h=np.nan; prev_l=np.nan
    state=0

    for i in range(n):
        if swing_high[i]:
            if np.isfinite(prev_h):
                if high[i]>prev_h: hh[i]=1
                elif high[i]<prev_h: lh[i]=1
            prev_h=high[i]
        if swing_low[i]:
            if np.isfinite(prev_l):
                if low[i]>prev_l: hl[i]=1
                elif low[i]<prev_l: ll[i]=1
            prev_l=low[i]
        last_high[i]=prev_h; last_low[i]=prev_l

        if hh[i] or hl[i]: state=1
        if lh[i] or ll[i]: state=-1
        trend[i]=state
    return last_high,last_low,hh,lh,hl,ll,trend


def add_market_structure_features(df:pl.DataFrame)->pl.DataFrame:
    from .swings import add_swing_features

    out=df.sort("timestamp")
    if "f_swing_high_candidate" not in out.columns:
        out=add_swing_features(out)
    sh=out["f_swing_high_candidate"].to_numpy()
    sl=out["f_swing_low_candidate"].to_numpy()
    last_high,last_low,hh,lh,hl,ll,trend=_state_arrays(
        out["high"].to_numpy(),out["low"].to_numpy(),sh.astype(bool),sl.astype(bool)
    )
    eps=1e-9
    out=out.with_columns(
        pl.Series("f_last_swing_high",last_high),
        pl.Series("f_last_swing_low",last_low),
        pl.Series("f_higher_high",hh,dtype=pl.Int8),
        pl.Series("f_lower_high",lh,dtype=pl.Int8),
        pl.Series("f_higher_low",hl,dtype=pl.Int8),
        pl.Series("f_lower_low",ll,dtype=pl.Int8),
        pl.Series("f_structure_trend",trend,dtype=pl.Int8),
    ).with_columns(
        (
            (pl.col("close")-pl.col("f_last_swing_high"))
            / (pl.col("f_last_swing_high")-pl.col("f_last_swing_low")+eps)
        ).alias("f_position_vs_structure_high"),
        (
            (pl.col("close")-pl.col("f_last_swing_low"))
            / (pl.col("f_last_swing_high")-pl.col("f_last_swing_low")+eps)
        ).alias("f_position_vs_structure_low"),
        (pl.col("close")>pl.col("f_last_swing_high")).cast(pl.Int8).alias("f_break_of_structure_up"),
        (pl.col("close")<pl.col("f_last_swing_low")).cast(pl.Int8).alias("f_break_of_structure_down"),
        (
            pl.col("close").rolling_mean(12)
            - pl.col("close").rolling_mean(48)
        ).alias("f_trend_context"),
        (
            pl.col("high").rolling_max(12)
            - pl.col("low").rolling_min(12)
        ).alias("f_short_structure_range"),
    )
    return out
