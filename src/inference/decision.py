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
    target_points:float=200.0,
    stop_points:float=70.0,
    min_probability:float=0.55,
    min_edge:float=0.10,
)->Decision:
    probs=np.asarray([p_short,p_none,p_long],dtype=float)
    if np.any(probs<0) or probs.sum()<=0:
        raise ValueError("probabilities must be non-negative and non-zero")
    probs/=probs.sum()

    short_ev=probs[0]*target_points-probs[2]*stop_points
    long_ev=probs[2]*target_points-probs[0]*stop_points
    directional_idx=0 if probs[0]>=probs[2] else 2
    directional=max(float(probs[0]),float(probs[2]))
    other=probs[2] if directional_idx==0 else probs[0]

    if directional<min_probability:
        return Decision(0,directional,0.0,"abstain_probability")
    if directional-float(probs[1])<min_edge:
        return Decision(0,directional,0.0,"abstain_vs_no_event")
    if directional-float(other)<min_edge:
        return Decision(0,directional,0.0,"abstain_directional_tie")

    if directional_idx==0:
        return Decision(-1,float(probs[0]),float(short_ev),"short")
    return Decision(1,float(probs[2]),float(long_ev),"long")
