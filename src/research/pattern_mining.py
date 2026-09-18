from __future__ import annotations
import numpy as np

def encode_direction_pattern(directions:np.ndarray,length:int)->int:
    d=np.asarray(directions,dtype=np.int64)[-length:]+1
    if len(d)<length: raise ValueError("not enough observations")
    value=0
    for v in d: value=value*3+int(v)
    return value

def rolling_pattern_ids(directions:np.ndarray,length:int)->np.ndarray:
    d=np.asarray(directions,dtype=np.int64)+1
    n=len(d)-length+1
    if n<=0:return np.array([],dtype=np.int64)
    base=3**np.arange(length-1,-1,-1,dtype=np.int64)
    return np.array([int(np.dot(d[i:i+length],base)) for i in range(n)],dtype=np.int64)
