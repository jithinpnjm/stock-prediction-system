import pandas as pd

from src.validation.purged_cv import CombinatorialPurgedCV, PurgedWalkForwardSplit


def test_walk_forward_is_chronological_and_purged():
    starts = pd.date_range("2026-01-01", periods=20, freq="5min", tz="UTC")
    ends = starts + pd.Timedelta(minutes=10)
    splitter = PurgedWalkForwardSplit(n_splits=3, embargo=pd.Timedelta(minutes=5))
    folds = list(splitter.split(starts, ends))
    assert folds
    for train_idx, test_idx in folds:
        assert train_idx.max() < test_idx.min()
        assert ends[train_idx].max() < starts[test_idx].min()


def test_cpcv_produces_multiple_paths():
    starts = pd.date_range("2026-01-01", periods=24, freq="5min", tz="UTC")
    ends = starts + pd.Timedelta(minutes=5)
    splitter = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    folds = list(splitter.split(starts, ends))
    assert len(folds) == 15
