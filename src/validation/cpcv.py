from __future__ import annotations

from itertools import combinations
import math

import numpy as np


def _embargo_value(embargo: int | np.integer | np.timedelta64) -> np.timedelta64 | int:
    return embargo


def combinatorial_purged_splits(
    timestamps: np.ndarray,
    event_end: np.ndarray,
    *,
    n_groups: int = 6,
    test_groups: int = 2,
    embargo: int = 0,
):
    """Generate CPCV splits with interval purge and symmetric embargo.

    Training events are retained on either side of the test groups, except
    when their information/outcome interval overlaps a test interval or falls
    inside the embargo window surrounding it.
    """
    ts = np.asarray(timestamps)
    end = np.asarray(event_end)
    if len(ts) != len(end):
        raise ValueError("timestamps and event_end must have equal length")
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    if not 1 <= test_groups < n_groups:
        raise ValueError("test_groups must be in [1, n_groups)")

    order = np.argsort(ts)
    groups = [g for g in np.array_split(order, n_groups) if len(g)]

    for ids in combinations(range(len(groups)), test_groups):
        test = np.sort(np.concatenate([groups[i] for i in ids]))
        train = np.sort(np.concatenate(
            [groups[i] for i in range(len(groups)) if i not in ids]
        ))

        keep = np.ones(len(train), dtype=bool)
        for idx in test:
            left = ts[idx] - embargo
            right = end[idx] + embargo
            keep &= ~(
                (ts[train] <= right)
                & (end[train] >= left)
            )

        yield train[keep], test


def cpcv_path_count(n_groups: int, test_groups: int) -> int:
    return math.comb(n_groups, test_groups)
