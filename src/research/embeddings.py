from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeatureEmbedder:
    def __init__(self, n_components: int = 16, seed: int = 42):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=seed)

    def fit(self, X: np.ndarray) -> FeatureEmbedder:
        self.pca.fit(self.scaler.fit_transform(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(X))
