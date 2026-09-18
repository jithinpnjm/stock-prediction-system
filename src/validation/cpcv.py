from __future__ import annotations

from itertools import combinations

import numpy as np


def combinatorial_splits(
    timestamps: np.ndarray,
    n_groups: int = 6,
    test_groups: int = 2,
):
    """Generate chronological CPCV train/test index combinations."""
    n = len(timestamps)
    if n_groups < 3 or not 1 <= test_groups < n_groups:
        raise ValueError("invalid CPCV group configuration")
    order = np.argsort(timestamps)
    groups = [g for g in np.array_split(order, n_groups) if len(g)]
    for test_group_ids in combinations(range(len(groups)), test_groups):
        test_set = np.concatenate([groups[i] for i in test_group_ids])
        train_set = np.concatenate(
            [groups[i] for i in range(len(groups)) if i not in test_group_ids]
        )
        yield np.sort(train_set), np.sort(test_set)


def cpcv_path_count(n_groups: int, test_groups: int) -> int:
    import math
    return math.comb(n_groups, test_groups)
