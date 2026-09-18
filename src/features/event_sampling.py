from __future__ import annotations

import numpy as np
import polars as pl


def cusum_events(
    returns: np.ndarray,
    threshold: float,
) -> np.ndarray:
    s_pos = 0.0
    s_neg = 0.0
    events = np.zeros(len(returns), dtype=np.int8)
    for i, r in enumerate(np.nan_to_num(returns, nan=0.0)):
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        if s_pos > threshold:
            events[i] = 1
            s_pos = 0.0
        elif s_neg < -threshold:
            events[i] = 1
            s_neg = 0.0
    return events


def add_event_sampling_features(
    df: pl.DataFrame,
    threshold_multiple: float = 2.0,
) -> pl.DataFrame:
    ret = df["close"].pct_change().to_numpy()
    scale = float(np.nanstd(ret))
    threshold = max(scale * threshold_multiple, 1e-6)
    events = cusum_events(ret, threshold)
    return df.with_columns(pl.Series("f_cusum_event", events, dtype=pl.Int8))
