"""
label_logic.py
---------------
Core, side-effect-free label computation shared by build_labeled_dataset.py
and its self-test. Kept in its own module so the leakage-sensitive math has
exactly one implementation and one place to test it.

Label definition (locked, see project plan):
  For candle t with close C_t, label = 1 if at ANY point over the next
  `horizon` candles (t+1 .. t+horizon, never including t itself), high
  touches >= C_t*(1+pct) OR low touches <= C_t*(1-pct). Else 0.

Computed strictly per trading session (calendar day) — a day's last
`horizon` rows never get a label, rather than silently looking across the
overnight/multi-day gap into the next session.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def label_session(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   horizon: int, pct: float) -> np.ndarray:
    """Return a float array of length len(close): 1.0 / 0.0 / NaN (undefined,
    not enough forward candles left in this session) per row."""
    n = len(close)
    label = np.full(n, np.nan)
    if n <= horizon:
        return label

    # window i (0-indexed) covers high[i : i+horizon] -> valid i in [0, n-horizon]
    fwd_max = sliding_window_view(high, horizon).max(axis=1)
    fwd_min = sliding_window_view(low, horizon).min(axis=1)

    valid_n = n - horizon  # rows t = 0 .. valid_n-1 get a label
    upper = close[:valid_n] * (1 + pct)
    lower = close[:valid_n] * (1 - pct)

    # row t's forward window (t+1 .. t+horizon) is fwd_max/fwd_min at index t+1
    window_max = fwd_max[1:valid_n + 1]
    window_min = fwd_min[1:valid_n + 1]

    label[:valid_n] = np.where((window_max >= upper) | (window_min <= lower), 1.0, 0.0)
    return label
