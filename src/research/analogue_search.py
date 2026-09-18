from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class AnalogueSearcher:
    """Nearest-neighbour search with an explicit as-of timestamp boundary."""

    def __init__(self, n_neighbors: int = 20):
        self.scaler = StandardScaler()
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.timestamps = None
        self.X = None
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray, timestamps: np.ndarray) -> "AnalogueSearcher":
        X = np.asarray(X, dtype=float)
        ts = np.asarray(timestamps)
        if len(X) != len(ts):
            raise ValueError("X and timestamps must align")
        self.X = self.scaler.fit_transform(X)
        self.nn.fit(self.X)
        self.timestamps = ts
        return self

    def query(
        self, X: np.ndarray, as_of_timestamp
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.X is None or self.timestamps is None:
            raise RuntimeError("searcher is not fitted")
        mask = self.timestamps < as_of_timestamp
        if not mask.any():
            raise ValueError("no historical analogues before as_of_timestamp")
        candidates = self.X[mask]
        query = self.scaler.transform(np.asarray(X, dtype=float))
        k = min(self.n_neighbors, len(candidates))
        distances = np.linalg.norm(
            candidates[None, :, :] - query[:, None, :], axis=2
        )
        idx = np.argpartition(distances, k - 1, axis=1)[:, :k]
        row = np.arange(len(query))[:, None]
        order = np.argsort(distances[row, idx], axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        original = np.flatnonzero(mask)[idx]
        return original, distances[row, idx]
