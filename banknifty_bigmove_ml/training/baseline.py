"""
baseline.py
------------
Reference models the promotion gate compares candidates against (Phase 2).
Both are "raw OHLCV only" — no hand-crafted indicators — just a much
simpler function class than the sequence models in models.py.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


def naive_predict(y_train: np.ndarray, n_val: int) -> np.ndarray:
    """Constant prediction = empirical positive rate of the training fold."""
    rate = float(np.mean(y_train))
    return np.full(n_val, rate)


def logreg_predict(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray) -> np.ndarray:
    """Flatten each (lookback, 4) window into a single vector and fit plain
    logistic regression — a linear baseline over the same raw features the
    sequence models see, just without any temporal structure."""
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    X_train_flat = X_train.reshape(n_train, -1)
    X_val_flat = X_val.reshape(n_val, -1)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train_flat, y_train)
    return clf.predict_proba(X_val_flat)[:, 1]
