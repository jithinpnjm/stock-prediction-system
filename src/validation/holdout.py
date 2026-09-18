from __future__ import annotations

import numpy as np


def chronological_holdout(
    timestamps: np.ndarray,
    *,
    holdout_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Return development and frozen-holdout indices in chronological order."""
    ts = np.asarray(timestamps)
    if not 0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be in (0, 0.5)")
    order = np.argsort(ts)
    cut = int(len(order) * (1.0 - holdout_fraction))
    if cut < 1 or cut >= len(order):
        raise ValueError("dataset too small for requested holdout")
    return order[:cut], order[cut:]


def verify_holdout_is_future(
    timestamps: np.ndarray,
    development_idx: np.ndarray,
    holdout_idx: np.ndarray,
) -> None:
    ts = np.asarray(timestamps)
    if ts[development_idx].max() >= ts[holdout_idx].min():
        raise ValueError("holdout is not strictly after development data")
