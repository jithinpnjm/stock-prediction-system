from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations

import numpy as np
import pandas as pd

DEFAULT_EMBARGO = pd.Timedelta(minutes=5)


def _to_ns(values: object) -> np.ndarray:
    return pd.to_datetime(values, utc=True).astype("int64").to_numpy()


class PurgedWalkForwardSplit:
    def __init__(self, n_splits: int = 5, *, embargo: pd.Timedelta | None = None) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        self.n_splits = n_splits
        self.embargo = DEFAULT_EMBARGO if embargo is None else embargo

    def split(self, event_start: object, event_end: object) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        starts = _to_ns(event_start)
        ends = _to_ns(event_end)
        if len(starts) != len(ends):
            raise ValueError("event_start and event_end lengths differ")
        n = len(starts)
        if n == 0:
            return
        block = max(1, n // (self.n_splits + 1))
        gap_ns = int(self.embargo.total_seconds() * 1e9)
        for i in range(self.n_splits):
            test_start_idx = (i + 1) * block
            test_end_idx = n if i == self.n_splits - 1 else min(n, (i + 2) * block)
            test_idx = np.arange(test_start_idx, test_end_idx, dtype=int)
            if len(test_idx) == 0:
                continue
            test_start_time = starts[test_idx[0]]
            train = np.arange(0, test_idx[0], dtype=int)
            train = train[ends[train] < test_start_time]
            train = train[starts[train] < test_start_time - gap_ns]
            if len(train):
                yield train, test_idx


class CombinatorialPurgedCV:
    def __init__(self, n_groups: int = 6, n_test_groups: int = 2, *, embargo: pd.Timedelta | None = None) -> None:
        if n_groups < 2:
            raise ValueError("n_groups must be >= 2")
        if n_test_groups <= 0 or n_test_groups >= n_groups:
            raise ValueError("n_test_groups must be between 1 and n_groups-1")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo = DEFAULT_EMBARGO if embargo is None else embargo

    def split(self, event_start: object, event_end: object) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        starts = _to_ns(event_start)
        ends = _to_ns(event_end)
        if len(starts) != len(ends):
            raise ValueError("event_start and event_end lengths differ")
        n = len(starts)
        groups = np.array_split(np.arange(n), self.n_groups)
        gap_ns = int(self.embargo.total_seconds() * 1e9)
        for selected in combinations(range(self.n_groups), self.n_test_groups):
            test_idx = np.concatenate([groups[i] for i in selected])
            test_idx.sort()
            test_left = starts[test_idx].min()
            test_right = ends[test_idx].max()
            keep = np.ones(n, dtype=bool)
            keep[test_idx] = False
            overlaps = (ends >= test_left) & (starts <= test_right + gap_ns)
            keep &= ~overlaps
            train_idx = np.flatnonzero(keep)
            if len(train_idx):
                yield train_idx, test_idx
