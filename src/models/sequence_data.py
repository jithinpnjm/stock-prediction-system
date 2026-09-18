from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SequenceBatch:
    X:np.ndarray
    y:np.ndarray
    timestamps:np.ndarray


def build_sequences(
    features:np.ndarray,
    labels:np.ndarray,
    timestamps:np.ndarray,
    *,
    sequence_length:int=48,
)->SequenceBatch:
    X=np.asarray(features,dtype=np.float32)
    y=np.asarray(labels)
    ts=np.asarray(timestamps)
    if len(X)!=len(y) or len(X)!=len(ts):
        raise ValueError("features, labels and timestamps must have equal length")
    if sequence_length<2 or len(X)<sequence_length:
        raise ValueError("not enough rows for requested sequence length")
    out_X=[]; out_y=[]; out_ts=[]
    for end in range(sequence_length-1,len(X)):
        window=X[end-sequence_length+1:end+1]
        if not np.all(np.isfinite(window)): continue
        out_X.append(window)
        out_y.append(y[end])
        out_ts.append(ts[end])
    if not out_X:
        raise ValueError("no finite sequences produced")
    return SequenceBatch(
        np.stack(out_X),np.asarray(out_y),np.asarray(out_ts)
    )
