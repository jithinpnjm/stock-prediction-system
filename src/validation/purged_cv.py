from __future__ import annotations

from collections.abc import Generator

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
