from __future__ import annotations

import numpy as np
import polars as pl


def cusum_events(
    returns: np.ndarray,
    thresholds: np.ndarray | float,
) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    t = (
        np.full(len(r), float(thresholds))
        if np.isscalar(thresholds)
        else np.asarray(thresholds, dtype=float)
    )
    pos = neg = 0.0
    events = np.zeros(len(r), dtype=np.int8)
    for i, x in enumerate(np.nan_to_num(r, nan=0.0)):
        threshold = max(float(t[i]), 1e-9)
        pos = max(0.0, pos + x)
        neg = min(0.0, neg + x)
        if pos > threshold:
            events[i] = 1
            pos = 0.0
            neg = 0.0
        elif neg < -threshold:
            events[i] = 1
            pos = 0.0
            neg = 0.0
    return events


def add_cusum_events(
    df: pl.DataFrame,
    threshold_atr: float = 4.0,
    atr_column: str = "atr_14",
) -> pl.DataFrame:
    """Causal CUSUM filter on price changes, reset at each session start.

    Accumulates signed close-to-close price changes and flags a bar
    where the running sum exceeds threshold_atr * atr_column. Both the
    delta and the accumulator reset at the first bar of each session
    so no information crosses a session boundary.
    """
    out = df.sort("timestamp")
    out = out.with_columns(pl.col("timestamp").dt.date().alias("_session_date"))
    delta = (pl.col("close") - pl.col("close").shift(1).over("_session_date")).fill_null(0.0)
    out = out.with_columns(delta.alias("_delta"))

    deltas = out["_delta"].to_numpy()
    sessions = out["_session_date"].to_numpy()
    atr = out[atr_column].to_numpy()
    threshold = (
        np.nan_to_num(atr, nan=np.nanmedian(atr) if np.isfinite(atr).any() else 1e-4)
        * threshold_atr
    )

    events = np.zeros(len(deltas), dtype=np.int8)
    pos = neg = 0.0
    prev_session = None
    for i, x in enumerate(deltas):
        if sessions[i] != prev_session:
            pos = neg = 0.0
            prev_session = sessions[i]
        t = max(float(threshold[i]), 1e-9)
        pos = max(0.0, pos + x)
        neg = min(0.0, neg + x)
        if pos > t or neg < -t:
            events[i] = 1
            pos = neg = 0.0

    return out.drop(["_session_date", "_delta"]).with_columns(
        pl.Series("cusum_event", events, dtype=pl.Int8)
    )


def add_event_sampling_features(
    df: pl.DataFrame,
    threshold_multiple: float = 2.0,
    volatility_lookback: int = 60,
) -> pl.DataFrame:
    out = df.sort("timestamp")
    if "f_return_1" not in out.columns:
        out = out.with_columns((pl.col("close") / pl.col("close").shift(1) - 1).alias("f_return_1"))
    ret = out["f_return_1"].to_numpy()
    scale = out["f_return_1"].rolling_std(volatility_lookback).shift(1).to_numpy()
    threshold = np.nan_to_num(scale, nan=np.nanmedian(scale) if np.isfinite(scale).any() else 1e-4)
    events = cusum_events(ret, threshold * threshold_multiple)
    return out.with_columns(
        pl.Series("f_cusum_event", events, dtype=pl.Int8),
        pl.Series("f_cusum_threshold", threshold * threshold_multiple),
    )
