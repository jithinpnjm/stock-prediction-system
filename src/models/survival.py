from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class DiscreteHazardModel:
    """Discrete-time hazard benchmark with explicit horizon-bucket state."""

    def __init__(self):
        self.models:dict[int,LogisticRegression]={}

    def fit(self,X:np.ndarray,event_buckets:np.ndarray,max_bucket:int):
        buckets=np.asarray(event_buckets,dtype=int)
        self.models={}
        for bucket in range(max_bucket+1):
            at_risk=buckets>=bucket
            if not at_risk.any(): continue
            y=(buckets[at_risk]==bucket).astype(int)
            if np.unique(y).size<2:
                continue
            model=LogisticRegression(max_iter=1000,class_weight="balanced")
            model.fit(X[at_risk],y)
            self.models[bucket]=model
        if not self.models: raise ValueError("no estimable hazard buckets")
        return self

    def predict_hazard(self,X:np.ndarray)->dict[int,np.ndarray]:
        if not self.models: raise RuntimeError("model is not fitted")
        return {b:m.predict_proba(X)[:,1] for b,m in self.models.items()}

    def predict_survival(self,X:np.ndarray)->np.ndarray:
        hazards=self.predict_hazard(X)
        survival=np.ones(len(X))
        for bucket in sorted(hazards):
            survival*=1.0-np.clip(hazards[bucket],0,1)
        return survival
