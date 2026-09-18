import numpy as np

from src.validation.cpcv import combinatorial_purged_splits, cpcv_path_count
from src.validation.purged_cv import PurgedTimeSeriesSplit


def test_walk_forward_is_chronological_and_purged():
    starts = np.arange(20, dtype=np.int64)
    ends = starts + 3
    splitter = PurgedTimeSeriesSplit(n_splits=3, embargo=2)
    folds = list(splitter.split(starts, ends))
    assert folds
    for train_idx, test_idx in folds:
        if len(train_idx) == 0:
            continue
        assert train_idx.max() < test_idx.min()
        assert np.all(ends[train_idx] < starts[test_idx].min() - 2)


def test_cpcv_produces_multiple_paths():
    starts = np.arange(24, dtype=np.int64)
    ends = starts + 1
    folds = list(
        combinatorial_purged_splits(
            starts,
            ends,
            n_groups=6,
            test_groups=2,
        )
    )
    assert len(folds) == cpcv_path_count(6, 2) == 15
