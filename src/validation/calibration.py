from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    def __init__(self, method: str = "sigmoid") -> None:
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError("method must be sigmoid or isotonic")
        self.method = method
        self.model = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y)
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip").fit(scores, y)
        else:
            self.model = LogisticRegression().fit(scores.reshape(-1, 1), y)
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("calibrator is not fitted")
        scores = np.asarray(scores, dtype=float)
        if self.method == "isotonic":
            p = np.asarray(self.model.predict(scores))
        else:
            p = self.model.predict_proba(scores.reshape(-1, 1))[:, 1]
        return np.column_stack([1 - p, p])


def expected_calibration_error(
    y_true: np.ndarray,
    p_positive: np.ndarray,
    bins: int = 10,
) -> float:
    y = np.asarray(y_true)
    p = np.clip(np.asarray(p_positive), 0.0, 1.0)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def brier_score(y_true: np.ndarray, p_positive: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_positive, dtype=float)
    return float(np.mean((p - y) ** 2))
