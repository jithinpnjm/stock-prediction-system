from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class AnalogueSearcher:
    def __init__(self, n_neighbors: int = 20):
        self.scaler = StandardScaler()
        self.index = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        self.matrix = None

    def fit(self, X: np.ndarray) -> "AnalogueSearcher":
        self.matrix = self.scaler.fit_transform(X)
        self.index.fit(self.matrix)
        return self

    def query(self, X_query: np.ndarray):
        if self.matrix is None:
            raise RuntimeError("fit must be called first")
        return self.index.kneighbors(self.scaler.transform(X_query))
