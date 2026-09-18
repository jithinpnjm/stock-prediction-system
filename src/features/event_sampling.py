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
    if len(t) != len(r):
        raise ValueError("thresholds must be scalar or match returns length")

    pos = 0.0
    neg = 0.0
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


def _session_cusum(
    returns: np.ndarray,
    threshold_multiple: float,
    volatility_lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    if volatility_lookback < 2:
        raise ValueError("volatility_lookback must be >= 2")

    scale = (
        pl.Series("r", returns)
        .rolling_std(volatility_lookback, min_samples=2)
        .shift(1)
        .fill_null(1e-4)
        .to_numpy()
    )
    scale = np.where(np.isfinite(scale), scale, 1e-4)
    threshold = scale * threshold_multiple
    return cusum_events(returns, threshold), threshold


def add_event_sampling_features(
    df: pl.DataFrame,
    threshold_multiple: float = 2.0,
    volatility_lookback: int = 60,
) -> pl.DataFrame:
    out = df.sort("timestamp").with_columns(
        pl.col("timestamp").dt.date().alias("_session_date"),
    )
    if "f_return_1" not in out.columns:
        out = out.with_columns(
            (
                pl.col("close")
                / pl.col("close").shift(1).over("_session_date")
                - 1
            ).alias("f_return_1")
        )

    events_parts: list[np.ndarray] = []
    threshold_parts: list[np.ndarray] = []
    for part in out.partition_by("_session_date", maintain_order=True):
        events, thresholds = _session_cusum(
            part["f_return_1"].to_numpy(),
            threshold_multiple,
            volatility_lookback,
        )
        events_parts.append(events)
        threshold_parts.append(thresholds)

    events = (
        np.concatenate(events_parts)
        if events_parts
        else np.empty(0, dtype=np.int8)
    )
    thresholds = (
        np.concatenate(threshold_parts)
        if threshold_parts
        else np.empty(0, dtype=float)
    )
    return out.with_columns(
        pl.Series("f_cusum_event", events, dtype=pl.Int8),
        pl.Series("f_cusum_threshold", thresholds),
    ).drop("_session_date")
