from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss


def calibrate_classifier(estimator, X, y, *, method: str = "sigmoid", cv=3):
    model = CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    model.fit(X, y)
    return model


def expected_calibration_error(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    probabilities = np.asarray(probabilities)
    y_true = np.asarray(y_true)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be shape [n_samples, n_classes]")
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (confidence > left) & (confidence <= right)
        if not mask.any():
            continue
        accuracy = np.mean(predicted[mask] == y_true[mask])
        ece += np.mean(mask) * abs(accuracy - np.mean(confidence[mask]))
    return float(ece)


def multiclass_logloss(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    return float(log_loss(y_true, probabilities, labels=np.arange(probabilities.shape[1])))
