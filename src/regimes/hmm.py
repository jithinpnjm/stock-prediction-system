from __future__ import annotations

import numpy as np


class GaussianHMMRegime:
    def __init__(self, n_regimes: int = 4, seed: int = 42):
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:
            raise ImportError(
                "Install hmmlearn to use GaussianHMMRegime"
            ) from exc
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=500,
            random_state=seed,
        )

    def fit(self, X: np.ndarray) -> "GaussianHMMRegime":
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
