from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


class MulticlassProbabilityCalibrator:
    """Calibrate probabilities using a temporally earlier OOF segment."""

    def __init__(self, seed: int = 42):
        self.model = LogisticRegression(C=1.0, max_iter=2000, multi_class="multinomial", random_state=seed)

    @staticmethod
    def _features(probabilities: np.ndarray) -> np.ndarray:
        p = np.asarray(probabilities, dtype=float)
        if p.ndim != 2:
            raise ValueError("probabilities must be 2D")
        p = np.clip(p, 1e-8, 1.0)
        p /= p.sum(axis=1, keepdims=True)
        return np.log(p)

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray):
        if len(probabilities) != len(y_true):
            raise ValueError("probability/label length mismatch")
        self.model.fit(self._features(probabilities), np.asarray(y_true))
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._features(probabilities))


def expected_calibration_error(probabilities: np.ndarray, y_true: np.ndarray, *, n_bins: int = 10) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(y_true)
    if p.ndim != 2:
        raise ValueError("probabilities must be 2D")
    if len(p) != len(y):
        raise ValueError("probability/label length mismatch")
    confidence = p.max(axis=1)
    predicted = p.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (confidence > left) & (confidence <= right)
        if mask.any():
            ece += np.mean(mask) * abs(np.mean(predicted[mask] == y[mask]) - np.mean(confidence[mask]))
    return float(ece)


def multiclass_logloss(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=float)
    return float(log_loss(np.asarray(y_true), p, labels=np.arange(p.shape[1])))


def time_split_oof(probabilities: np.ndarray, y_true: np.ndarray, *, calibration_fraction: float = 0.5):
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")
    split = int(len(y_true) * calibration_fraction)
    if split < 20 or len(y_true) - split < 20:
        raise ValueError("Not enough OOF rows for calibration")
    calibrator = MulticlassProbabilityCalibrator()
    calibrator.fit(probabilities[:split], y_true[:split])
    return calibrator, probabilities[split:], np.asarray(y_true)[split:]
