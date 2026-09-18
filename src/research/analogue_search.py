from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class AnalogueSearcher:
    def __init__(self,n_neighbors:int=20):
        self.scaler=StandardScaler(); self.nn=NearestNeighbors(n_neighbors=n_neighbors)
        self.fitted=False
    def fit(self,X:np.ndarray):
        self.nn.fit(self.scaler.fit_transform(X)); self.fitted=True; return self
    def query(self,X:np.ndarray):
        if not self.fitted: raise RuntimeError("searcher is not fitted")
        return self.nn.kneighbors(self.scaler.transform(X))
