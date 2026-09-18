from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


class TimeToEventModel:
    """Practical baseline for time-to-barrier research.

    The target is log(1 + duration_minutes). Censored observations are retained
    with a configurable lower-bound treatment and must be evaluated separately
    from uncensored events.
    """

    def __init__(self, seed: int = 42):
        self.model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=seed,
        )

    def fit(self, X, duration_minutes):
        target = np.log1p(np.asarray(duration_minutes, dtype=float))
        self.model.fit(X, target)
        return self

    def predict_minutes(self, X):
        return np.expm1(self.model.predict(X))
