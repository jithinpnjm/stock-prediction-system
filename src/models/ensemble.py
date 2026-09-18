from __future__ import annotations

import numpy as np


def weighted_probability_ensemble(
    probabilities: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    if not probabilities:
        raise ValueError("at least one probability matrix is required")
    if weights is None:
        weights=[1.0]*len(probabilities)
    if len(weights)!=len(probabilities):
        raise ValueError("weights/probabilities length mismatch")
    w=np.asarray(weights,dtype=float)
    if np.any(w<0) or w.sum()<=0:
        raise ValueError("weights must be non-negative and non-zero")
    w/=w.sum()
    out=sum(p*weight for p,weight in zip(probabilities,w))
    return out/out.sum(axis=1,keepdims=True)


def decision_with_abstention(
    probabilities: np.ndarray,
    *,
    class_labels: tuple[int,int,int]=(-1,0,1),
    min_probability: float = 0.55,
    min_edge: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    p=np.asarray(probabilities,dtype=float)
    idx=np.argmax(p,axis=1)
    best=p[np.arange(len(p)),idx]
    runner=np.partition(p,-2,axis=1)[:,-2]
    signals=np.zeros(len(p),dtype=np.int8)
    take=(best>=min_probability)&((best-runner)>=min_edge)
    signals[take]=np.asarray(class_labels,dtype=np.int8)[idx[take]]
    return signals,best
