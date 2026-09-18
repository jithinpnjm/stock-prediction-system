from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class DiscreteHazardModel:
    """Simple discrete-time hazard benchmark.

    It estimates P(event in a horizon | no event before horizon) per bucket.
    """

    def __init__(self):
        self.models: list[LogisticRegression] = []

    def fit(self, X: np.ndarray, event_buckets: np.ndarray, max_bucket: int):
        self.models=[]
        for bucket in range(max_bucket+1):
            at_risk=event_buckets >= bucket
            if at_risk.sum() == 0:
                continue
            y=(event_buckets[at_risk] == bucket).astype(int)
            model=LogisticRegression(max_iter=1000,class_weight="balanced")
            model.fit(X[at_risk],y)
            self.models.append(model)
        return self

    def predict_survival(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("model is not fitted")
        survival=np.ones(len(X))
        for model in self.models:
            hazard=model.predict_proba(X)[:,1]
            survival*=1.0-hazard
        return survival
