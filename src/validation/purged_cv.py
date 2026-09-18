from __future__ import annotations

from itertools import combinations
from typing import Iterator

import numpy as np
import pandas as pd


def _to_ns(values: object) -> np.ndarray:
    return pd.to_datetime(values, utc=True).astype("int64").to_numpy()


class PurgedWalkForwardSplit:
    def __init__(
        self,
        n_splits: int = 5,
        *,
        embargo: pd.Timedelta = pd.Timedelta(minutes=5),
        test_fraction: float | None = None,
    ) -> None:
        self.n_splits = n_splits
        self.embargo = embargo
        self.test_fraction = test_fraction

    def split(
        self,
        event_start: object,
        event_end: object,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        starts = _to_ns(event_start)
        ends = _to_ns(event_end)
        n = len(starts)
        if n == 0:
            return
        block = max(1, n // (self.n_splits + 1))
        for i in range(self.n_splits):
            test_start = (i + 1) * block
            test_end = n if i == self.n_splits - 1 else min(n, (i + 2) * block)
            test_idx = np.arange(test_start, test_end, dtype=int)
            if len(test_idx) == 0:
                continue
            cutoff = starts[test_idx[0]]
            test_right = starts[test_idx[-1]]
            gap_ns = int(self.embargo.total_seconds() * 1e9)
            train = np.arange(0, test_idx[0], dtype=int)
            train = train[ends[train] < cutoff]
            train = train[starts[train] < cutoff - gap_ns]
            # No post-test training samples are used in walk-forward CV.
            _ = test_right
            if len(train):
                yield train, test_idx


class CombinatorialPurgedCV:
    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        *,
        embargo: pd.Timedelta = pd.Timedelta(minutes=5),
    ) -> None:
        if n_test_groups <= 0 or n_test_groups >= n_groups:
            raise ValueError("n_test_groups must be between 1 and n_groups-1")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo = embargo

    def split(
        self,
        event_start: object,
        event_end: object,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        starts = _to_ns(event_start)
        ends = _to_ns(event_end)
        n = len(starts)
        groups = np.array_split(np.arange(n), self.n_groups)
        gap_ns = int(self.embargo.total_seconds() * 1e9)

        for selected in combinations(range(self.n_groups), self.n_test_groups):
            test_idx = np.concatenate([groups[i] for i in selected])
            test_idx.sort()
            test_start = starts[test_idx].min()
            test_end = starts[test_idx].max()
            mask = np.ones(n, dtype=bool)
            mask[test_idx] = False

            # Purge any event whose information interval overlaps the test interval.
            mask &= ends < test_start
            # Also remove events in the embargo window immediately before the test.
            mask &= starts < (test_start - gap_ns)
            # And remove events after the test block's end.
            mask &= starts > (test_end + gap_ns)
            train_idx = np.flatnonzero(mask)

            if len(train_idx):
                yield train_idx, test_idx
