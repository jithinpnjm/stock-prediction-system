from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.contracts import MARKET_TIMEZONE


@dataclass(frozen=True)
class SequenceBatch:
    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    event_end: np.ndarray


def _session_dates(timestamps: np.ndarray) -> np.ndarray:
    local = pd.to_datetime(timestamps, utc=True).tz_convert(MARKET_TIMEZONE)
    return np.asarray(local.date)


def build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    event_end: np.ndarray | None = None,
    *,
    sequence_length: int = 48,
    session_dates: np.ndarray | None = None,
) -> SequenceBatch:
    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    ts = np.asarray(timestamps)
    end = ts if event_end is None else np.asarray(event_end)
    dates = (
        _session_dates(ts)
        if session_dates is None
        else np.asarray(session_dates)
    )

    if len(X) != len(y) or len(X) != len(ts) or len(X) != len(end) or len(X) != len(dates):
        raise ValueError("features, labels, timestamps, event_end and session_dates must match")
    if sequence_length < 2 or len(X) < sequence_length:
        raise ValueError("not enough rows for requested sequence")

    out_X: list[np.ndarray] = []
    out_y: list[object] = []
    out_ts: list[object] = []
    out_end: list[object] = []

    for i in range(sequence_length - 1, len(X)):
        start = i - sequence_length + 1
        if np.any(dates[start : i + 1] != dates[i]):
            continue

        window_ts = ts[start : i + 1]
        if len(window_ts) > 1:
            delta = np.diff(
                pd.to_datetime(window_ts, utc=True).astype("int64").to_numpy()
            )
            if not np.all(delta == 5 * 60 * 1_000_000_000):
                continue

        window = X[start : i + 1]
        if not np.all(np.isfinite(window)):
            continue

        out_X.append(window)
        out_y.append(y[i])
        out_ts.append(ts[i])
        out_end.append(end[i])

    if not out_X:
        raise ValueError("no valid finite in-session sequences produced")

    return SequenceBatch(
        np.stack(out_X),
        np.asarray(out_y),
        np.asarray(out_ts),
        np.asarray(out_end),
    )
