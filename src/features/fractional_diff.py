from __future__ import annotations

import numpy as np
import polars as pl


def fractional_weights(d: float, threshold: float = 1e-5, max_lags: int = 200) -> np.ndarray:
    weights=[1.0]
    for k in range(1,max_lags+1):
        w=-weights[-1]*(d-k+1)/k
        weights.append(w)
        if abs(w)<threshold:
            break
    return np.asarray(weights[::-1],dtype=float)


def frac_diff(values:np.ndarray,d:float,threshold:float=1e-5,max_lags:int=200)->np.ndarray:
    x=np.asarray(values,dtype=float)
    w=fractional_weights(d,threshold,max_lags)
    out=np.full(len(x),np.nan)
    width=len(w)
    for i in range(width-1,len(x)):
        window=x[i-width+1:i+1]
        if np.all(np.isfinite(window)):
            out[i]=float(np.dot(w,window))
    return out


def add_fractional_diff_feature(
    df:pl.DataFrame,
    column:str="close",
    d:float=0.4,
    threshold:float=1e-5,
    max_lags:int=200,
)->pl.DataFrame:
    values=frac_diff(df[column].to_numpy(),d,threshold,max_lags)
    return df.with_columns(
        pl.Series(f"f_fracdiff_{column}_{d:g}",values)
    )
