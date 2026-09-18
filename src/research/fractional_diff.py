from __future__ import annotations

import numpy as np
import pandas as pd


def fractional_difference_weights(
    d: float,
    *,
    threshold: float = 1e-5,
    max_lag: int = 10_000,
) -> np.ndarray:
    if not 0.0 <= d <= 1.0:
        raise ValueError("d must be in [0, 1]")
    weights = [1.0]
    for k in range(1, max_lag + 1):
        weight = -weights[-1] * (d - k + 1.0) / k
        weights.append(weight)
        if abs(weight) < threshold:
            break
    return np.asarray(weights, dtype=float)


def fractional_difference(
    series: pd.Series | np.ndarray,
    d: float,
    *,
    threshold: float = 1e-5,
) -> pd.Series:
    values = pd.Series(series, dtype="float64")
    weights = fractional_difference_weights(d, threshold=threshold)
    out = np.full(len(values), np.nan, dtype=float)
    x = values.to_numpy()

    for i in range(len(x)):
        width = min(i + 1, len(weights))
        window = x[i - width + 1 : i + 1]
        w = weights[:width][::-1]
        if np.isfinite(window).all():
            out[i] = float(np.dot(w, window))
    return pd.Series(out, index=values.index, name=values.name)
