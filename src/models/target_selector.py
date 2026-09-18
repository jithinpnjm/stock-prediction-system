from __future__ import annotations


def choose_target(
    side_probability_by_target: dict[float, float],
    *,
    stop_points: float = 70.0,
    min_probability: float = 0.55,
) -> float | None:
    """Choose the largest target with positive estimated point expectancy."""
    candidates = []
    for target, probability in sorted(side_probability_by_target.items()):
        p = float(probability)
        if p < min_probability:
            continue
        expected_value = p * float(target) - (1.0 - p) * stop_points
        if expected_value > 0:
            candidates.append(float(target))
    return max(candidates) if candidates else None
