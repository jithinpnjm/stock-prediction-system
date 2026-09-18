from __future__ import annotations

import numpy as np


def blend_probabilities(
    probability_arrays: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    if not probability_arrays:
        raise ValueError("At least one probability array is required")
    shapes = {np.asarray(p).shape for p in probability_arrays}
    if len(shapes) != 1:
        raise ValueError("All probability arrays must have the same shape")
    if weights is None:
        weights = [1.0] * len(probability_arrays)
    if len(weights) != len(probability_arrays):
        raise ValueError("weights length mismatch")
    w = np.asarray(weights, dtype=float)
    w /= w.sum()
    blended = sum(weight * np.asarray(p) for weight, p in zip(w, probability_arrays))
    blended /= blended.sum(axis=1, keepdims=True)
    return blended


def decision_from_probabilities(
    probabilities: np.ndarray,
    *,
    class_order=(-1, 0, 1),
    min_confidence: float = 0.5,
    min_edge: float = 0.1,
) -> np.ndarray:
    p = np.asarray(probabilities)
    order = np.asarray(class_order)
    top = p.argmax(axis=1)
    second = np.partition(p, -2, axis=1)[:, -2]
    confidence = p[np.arange(len(p)), top]
    edge = confidence - second
    decisions = np.zeros(len(p), dtype=int)
    valid = (confidence >= min_confidence) & (edge >= min_edge)
    decisions[valid] = order[top[valid]]
    return decisions


def expected_value_per_point(
    probabilities: np.ndarray,
    *,
    target_points: float,
    stop_points: float,
    class_order=(-1, 0, 1),
) -> np.ndarray:
    p = np.asarray(probabilities)
    mapping = {c: i for i, c in enumerate(class_order)}
    p_long = p[:, mapping[1]]
    p_short = p[:, mapping[-1]]
    return (
        p_long * (target_points + stop_points) - stop_points
    ), (
        p_short * (target_points + stop_points) - stop_points
    )
