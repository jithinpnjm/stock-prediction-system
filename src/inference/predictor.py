from __future__ import annotations

import numpy as np

from .decision import Decision,decide


def decision_from_probability_row(
    probabilities:np.ndarray,
    *,
    target_points:float=200,
    stop_points:float=70,
    min_probability:float=0.55,
    min_edge:float=0.10,
)->Decision:
    p=np.asarray(probabilities,dtype=float)
    if p.shape!=(3,):
        raise ValueError("probabilities must have shape (3,)")
    return decide(
        float(p[0]),float(p[1]),float(p[2]),
        target_points=target_points,stop_points=stop_points,
        min_probability=min_probability,min_edge=min_edge
    )
