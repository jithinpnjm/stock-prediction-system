from __future__ import annotations

import numpy as np
import pandas as pd


def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    timestamps,
    *,
    window: int = 32,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    ts = pd.to_datetime(timestamps, utc=True)
    sequences: list[np.ndarray] = []
    labels: list[object] = []
    ends: list[object] = []

    for end in range(window - 1, len(X), step):
        start = end - window + 1
        # Require contiguous 5m observations from the same session.
        dt = ts[start : end + 1].view("int64")
        if len(dt) > 1 and not np.all(np.diff(dt) == 5 * 60 * 1_000_000_000):
            continue
        sequences.append(X[start : end + 1])
        labels.append(y[end])
        ends.append(ts[end])
    if not sequences:
        return (
            np.empty((0, window, X.shape[1]), dtype=np.float32),
            np.empty((0,)),
            np.empty((0,), dtype=object),
        )
    return np.stack(sequences), np.asarray(labels), np.asarray(ends)


def chronological_sequence_split(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    *,
    validation_fraction: float = 0.2,
):
    cut = int(len(X_seq) * (1.0 - validation_fraction))
    return X_seq[:cut], y_seq[:cut], X_seq[cut:], y_seq[cut:]
