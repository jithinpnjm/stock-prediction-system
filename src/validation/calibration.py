from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    def __init__(self, method: str = "sigmoid") -> None:
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError("method must be sigmoid or isotonic")
        self.method = method
        self.model = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        scores=np.asarray(scores,dtype=float)
        y=np.asarray(y)
        if self.method=="isotonic":
            self.model=IsotonicRegression(out_of_bounds="clip").fit(scores,y)
        else:
            self.model=LogisticRegression(max_iter=2000).fit(scores.reshape(-1,1),y)
        return self

    def predict_proba(self,scores:np.ndarray)->np.ndarray:
        if self.model is None: raise RuntimeError("calibrator is not fitted")
        scores=np.asarray(scores,dtype=float)
        if self.method=="isotonic":
            p=np.asarray(self.model.predict(scores))
        else:
            p=self.model.predict_proba(scores.reshape(-1,1))[:,1]
        return np.column_stack([1-p,p])


class TemperatureScaler:
    """Multiclass temperature scaling fit only on a calibration/OOS set."""

    def __init__(self) -> None:
        self.temperature=1.0

    def fit(self, probabilities:np.ndarray, y:np.ndarray)->"TemperatureScaler":
        p=np.clip(np.asarray(probabilities,dtype=float),1e-8,1.0)
        p=p/p.sum(axis=1,keepdims=True)
        y=np.asarray(y,dtype=int)
        logp=np.log(p)

        def loss(log_t:float)->float:
            t=float(np.exp(log_t))
            z=logp/t
            z-=z.max(axis=1,keepdims=True)
            q=np.exp(z); q/=q.sum(axis=1,keepdims=True)
            idx=np.asarray(y)+1
            return float(-np.mean(np.log(np.clip(q[np.arange(len(y)),idx],1e-12,1.0))))

        result=minimize_scalar(loss,bounds=(-2.0,2.0),method="bounded")
        self.temperature=float(np.exp(result.x))
        return self

    def predict_proba(self,probabilities:np.ndarray)->np.ndarray:
        p=np.clip(np.asarray(probabilities,dtype=float),1e-8,1.0)
        p=p/p.sum(axis=1,keepdims=True)
        z=np.log(p)/self.temperature
        z-=z.max(axis=1,keepdims=True)
        q=np.exp(z)
        return q/q.sum(axis=1,keepdims=True)


def expected_calibration_error(y_true:np.ndarray,p_positive:np.ndarray,bins:int=10)->float:
    y=np.asarray(y_true,dtype=float)
    p=np.clip(np.asarray(p_positive,dtype=float),0,1)
    edges=np.linspace(0,1,bins+1)
    ece=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(p>=lo)&(p<=hi if hi==1 else p<hi)
        if mask.any(): ece+=float(mask.mean()*abs(y[mask].mean()-p[mask].mean()))
    return float(ece)


def brier_score(y_true:np.ndarray,p_positive:np.ndarray)->float:
    y=np.asarray(y_true,dtype=float); p=np.asarray(p_positive,dtype=float)
    return float(np.mean((p-y)**2))
