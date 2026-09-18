from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SequenceBatch:
    X:np.ndarray
    y:np.ndarray
    timestamps:np.ndarray
    event_end:np.ndarray


def build_sequences(
    features:np.ndarray,
    labels:np.ndarray,
    timestamps:np.ndarray,
    event_end:np.ndarray|None=None,
    *,
    sequence_length:int=48,
)->SequenceBatch:
    X=np.asarray(features,dtype=np.float32)
    y=np.asarray(labels)
    ts=np.asarray(timestamps)
    end=ts if event_end is None else np.asarray(event_end)
    if len(X)!=len(y) or len(X)!=len(ts) or len(X)!=len(end):
        raise ValueError("features, labels, timestamps and event_end must match")
    if sequence_length<2 or len(X)<sequence_length:
        raise ValueError("not enough rows for requested sequence")
    out_X=[]; out_y=[]; out_ts=[]; out_end=[]
    for i in range(sequence_length-1,len(X)):
        window=X[i-sequence_length+1:i+1]
        if not np.all(np.isfinite(window)): continue
        out_X.append(window); out_y.append(y[i]); out_ts.append(ts[i]); out_end.append(end[i])
    if not out_X: raise ValueError("no finite sequences produced")
    return SequenceBatch(
        np.stack(out_X),np.asarray(out_y),np.asarray(out_ts),np.asarray(out_end)
    )
