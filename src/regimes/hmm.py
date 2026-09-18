from __future__ import annotations
import numpy as np
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM=None

class GaussianRegimeHMM:
    def __init__(self,n_regimes:int=4,seed:int=42):
        if GaussianHMM is None:
            raise RuntimeError("Install the research extra to use HMM regimes")
        self.model=GaussianHMM(n_components=n_regimes,covariance_type="diag",n_iter=200,random_state=seed)
    def fit(self,X:np.ndarray):
        self.model.fit(X); return self
    def predict_proba(self,X:np.ndarray)->np.ndarray:
        return self.model.predict_proba(X)
    def predict(self,X:np.ndarray)->np.ndarray:
        return self.model.predict(X)
