from __future__ import annotations

from collections.abc import Generator
from itertools import combinations
from math import comb

import numpy as np


class PurgedTimeSeriesSplit:
    """Chronological CV with interval-aware purging and embargo.

    Each observation has [event_start, event_end]. A training event is removed
    when its information/outcome interval overlaps the test interval.
    """

    def __init__(self, n_splits: int = 5, embargo: int = 0) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.embargo = embargo

    def split(
        self,
        timestamps: np.ndarray,
        event_end: np.ndarray | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        n = len(timestamps)
        if n == 0:
            return
        ts = np.asarray(timestamps)
        end = ts if event_end is None else np.asarray(event_end)
        order = np.argsort(ts)
        ordered_ts = ts[order]
        ordered_end = end[order]
        edges = np.linspace(0, n, self.n_splits + 1, dtype=int)

        for fold in range(self.n_splits):
            test_lo, test_hi = edges[fold], edges[fold + 1]
            test_idx = order[test_lo:test_hi]
            if len(test_idx) == 0:
                continue
            test_start = ordered_ts[test_lo]
            test_end = ordered_ts[test_hi - 1]
            eligible = np.arange(0, test_lo)
            if self.embargo:
                cutoff = test_start - self.embargo
                eligible = eligible[ordered_end[eligible] < cutoff]
            else:
                eligible = eligible[ordered_end[eligible] < test_start]
            yield order[eligible], test_idx

    def get_n_splits(self) -> int:
        return self.n_splits


class PurgedWalkForwardSplit:
    """Expanding-window walk-forward CV with interval-aware purging/embargo.

    Unlike PurgedTimeSeriesSplit (which treats every block as an
    independent test fold), each fold's training set is every block
    strictly before that fold's test block, chronologically. The first
    block is never used as a test fold since it would have no training
    history.
    """

    def __init__(self, n_splits: int = 5, embargo: int = 0) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.embargo = embargo

    def split(
        self,
        timestamps: np.ndarray,
        event_end: np.ndarray | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        n = len(timestamps)
        if n == 0:
            return
        ts = np.asarray(timestamps)
        end = ts if event_end is None else np.asarray(event_end)
        order = np.argsort(ts)
        ordered_ts = ts[order]
        ordered_end = end[order]
        edges = np.linspace(0, n, self.n_splits + 1, dtype=int)

        for fold in range(1, self.n_splits):
            test_lo, test_hi = edges[fold], edges[fold + 1]
            test_idx = order[test_lo:test_hi]
            if len(test_idx) == 0:
                continue
            test_start = ordered_ts[test_lo]
            eligible = np.arange(0, test_lo)
            if self.embargo:
                cutoff = test_start - self.embargo
                eligible = eligible[ordered_end[eligible] < cutoff]
            else:
                eligible = eligible[ordered_end[eligible] < test_start]
            if len(eligible) == 0:
                continue
            yield order[eligible], test_idx

    def get_n_splits(self) -> int:
        return self.n_splits - 1


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (Lopez de Prado).

    The timeline is split into n_groups contiguous blocks. Every
    combination of n_test_groups blocks becomes one test path; the
    remaining blocks form the training set after purging any training
    observation whose event interval overlaps an embargoed window
    around any of the selected test blocks. This yields C(n_groups,
    n_test_groups) train/test paths instead of a single walk-forward
    sequence, which is what lets CPCV support PBO/DSR-style robustness
    analysis over many resampled paths.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        embargo: int = 0,
    ) -> None:
        if n_groups < 2:
            raise ValueError("n_groups must be >= 2")
        if not (0 < n_test_groups < n_groups):
            raise ValueError("n_test_groups must be between 1 and n_groups - 1")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo = embargo

    def split(
        self,
        timestamps: np.ndarray,
        event_end: np.ndarray | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        n = len(timestamps)
        if n == 0:
            return
        ts = np.asarray(timestamps)
        end = ts if event_end is None else np.asarray(event_end)
        order = np.argsort(ts)
        ordered_ts = ts[order]
        ordered_end = end[order]
        edges = np.linspace(0, n, self.n_groups + 1, dtype=int)
        group_pos = [np.arange(edges[i], edges[i + 1]) for i in range(self.n_groups)]

        for test_groups in combinations(range(self.n_groups), self.n_test_groups):
            test_pos = np.sort(np.concatenate([group_pos[g] for g in test_groups]))
            if len(test_pos) == 0:
                continue
            train_pos = np.array(
                [p for g in range(self.n_groups) if g not in test_groups for p in group_pos[g]],
                dtype=int,
            )
            if len(train_pos) == 0:
                continue
            purge_mask = np.zeros(len(train_pos), dtype=bool)
            for g in test_groups:
                gpos = group_pos[g]
                if len(gpos) == 0:
                    continue
                lo = ordered_ts[gpos].min()
                hi = ordered_ts[gpos].max()
                if self.embargo:
                    lo = lo - self.embargo
                    hi = hi + self.embargo
                purge_mask |= (ordered_end[train_pos] >= lo) & (ordered_ts[train_pos] <= hi)
            eligible = train_pos[~purge_mask]
            if len(eligible) == 0:
                continue
            yield order[eligible], order[test_pos]

    def get_n_splits(self) -> int:
        return comb(self.n_groups, self.n_test_groups)
