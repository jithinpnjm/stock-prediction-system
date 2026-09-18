from __future__ import annotations

import numpy as np


def weighted_probability_ensemble(
    probabilities: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    if not probabilities:
        raise ValueError("at least one probability matrix is required")
    if weights is None:
        weights = [1.0] * len(probabilities)
    if len(weights) != len(probabilities):
        raise ValueError("weights/probabilities length mismatch")
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative and non-zero")
    w /= w.sum()
    out = sum(p * weight for p, weight in zip(probabilities, w))
    return out / out.sum(axis=1, keepdims=True)


def decision_from_probabilities(
    probabilities: np.ndarray,
    *,
    class_labels: tuple[int, int, int] = (-1, 0, 1),
    min_confidence: float = 0.55,
    min_edge: float = 0.10,
) -> np.ndarray:
    """Trade signal (-1/0/1) from [p_short, p_flat, p_long] probabilities.

    Thin wrapper around decision_with_abstention that returns only the
    signal array, matching src.backtest.decision's usage.
    """
    signals, _ = decision_with_abstention(
        probabilities,
        class_labels=class_labels,
        min_probability=min_confidence,
        min_edge=min_edge,
    )
    return signals


def expected_value_per_point(
    probabilities: np.ndarray,
    *,
    target_points: float,
    stop_points: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected P&L in points for taking a long/short trade.

    probabilities columns are [p_short, p_flat, p_long], i.e. the
    probability the price hits the short barrier first, times out, or
    hits the long barrier first. For a long entry, the opposite
    (short) barrier is the trade's stop-out; for a short entry the
    long barrier is the stop-out. A timeout is treated as ~0 P&L.
    """
    p = np.asarray(probabilities, dtype=float)
    p_short, _p_flat, p_long = p[:, 0], p[:, 1], p[:, 2]
    ev_long = p_long * target_points - p_short * stop_points
    ev_short = p_short * target_points - p_long * stop_points
    return ev_long, ev_short


def decision_with_abstention(
    probabilities: np.ndarray,
    *,
    class_labels: tuple[int, int, int] = (-1, 0, 1),
    min_probability: float = 0.55,
    min_edge: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probabilities, dtype=float)
    idx = np.argmax(p, axis=1)
    best = p[np.arange(len(p)), idx]
    runner = np.partition(p, -2, axis=1)[:, -2]
    signals = np.zeros(len(p), dtype=np.int8)
    take = (best >= min_probability) & ((best - runner) >= min_edge)
    signals[take] = np.asarray(class_labels, dtype=np.int8)[idx[take]]
    return signals, best
