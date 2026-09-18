from __future__ import annotations
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class RegimeClusterer:
    def __init__(self,n_regimes:int=4,seed:int=42):
        self.scaler=StandardScaler()
        self.model=GaussianMixture(n_components=n_regimes,random_state=seed,n_init=5)
    def fit(self,X:np.ndarray):
        self.model.fit(self.scaler.fit_transform(X)); return self
    def predict_proba(self,X:np.ndarray)->np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))
    def predict(self,X:np.ndarray)->np.ndarray:
        return self.model.predict(self.scaler.transform(X))
