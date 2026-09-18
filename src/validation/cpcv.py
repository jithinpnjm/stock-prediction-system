from __future__ import annotations

from itertools import combinations

import numpy as np


def combinatorial_purged_splits(
    timestamps:np.ndarray,
    event_end:np.ndarray,
    *,
    n_groups:int=6,
    test_groups:int=2,
    embargo:int=0,
):
    """Generate CPCV splits with event-interval purge and embargo."""
    order=np.argsort(timestamps)
    groups=[g for g in np.array_split(order,n_groups) if len(g)]
    ts=np.asarray(timestamps); end=np.asarray(event_end)
    for ids in combinations(range(len(groups)),test_groups):
        test=np.sort(np.concatenate([groups[i] for i in ids]))
        test_start=ts[test].min()
        train=np.sort(np.concatenate([groups[i] for i in range(len(groups)) if i not in ids]))
        cutoff=test_start-embargo
        train=train[end[train]<cutoff]
        yield train,test


def cpcv_path_count(n_groups:int,test_groups:int)->int:
    import math
    return math.comb(n_groups,test_groups)
