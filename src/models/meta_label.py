from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class MetaLabeler:
    """Learns whether a proposed base-model trade should be taken."""

    def __init__(self, seed: int = 42):
        self.model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )

    def fit(self, meta_features: np.ndarray, outcome: np.ndarray) -> "MetaLabeler":
        self.model.fit(meta_features, outcome)
        return self

    def predict_proba(self, meta_features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(meta_features)[:, 1]

    def predict(self, meta_features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(meta_features) >= threshold).astype(int)
