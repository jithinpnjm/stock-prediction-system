from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Decision:
    signal:int
    probability:float
    expected_value_points:float
    reason:str


def decide(
    p_short:float,p_none:float,p_long:float,
    *,
    target_points:float=200,
    stop_points:float=70,
    min_probability:float=0.55,
    min_edge:float=0.10,
)->Decision:
    probs=np.asarray([p_short,p_none,p_long],dtype=float)
    if np.any(probs<0) or probs.sum()<=0:
        raise ValueError("probabilities must be non-negative and non-zero")
    probs/=probs.sum()
    long_ev=probs[2]*target_points-(1.0-probs[2])*stop_points
    short_ev=probs[0]*target_points-(1.0-probs[0])*stop_points
    idx=int(np.argmax(probs[[0,2]]))
    best=float(max(probs[0],probs[2]))
    other=float(min(probs[0],probs[2]))
    if best<min_probability or best-other<min_edge:
        return Decision(0,best,0.0,"abstain")
    if idx==0:
        return Decision(-1,float(probs[0]),float(short_ev),"short")
    return Decision(1,float(probs[2]),float(long_ev),"long")
