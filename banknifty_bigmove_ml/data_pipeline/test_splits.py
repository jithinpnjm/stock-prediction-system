import pandas as pd
import pytest

from splits import build_walk_forward_folds, slice_fold, slice_holdout


def _days(n):
    return pd.date_range("2021-01-01", periods=n, freq="B")  # business days


def test_holdout_absorbs_full_remainder_not_capped_at_one_block():
    # 1242 days, 400 initial train, 100-day blocks -> 7 folds, remainder > 100 days
    days = _days(1242)
    plan = build_walk_forward_folds(days, initial_train_days=400, val_block_days=100)
    assert len(plan.folds) == 7
    assert plan.holdout_end == days[-1]  # every trailing day is covered, none dropped


def test_folds_are_expanding_and_non_overlapping():
    days = _days(1242)
    plan = build_walk_forward_folds(days, initial_train_days=400, val_block_days=100)
    for i, fold in enumerate(plan.folds):
        assert fold.train_start == days[0]
        if i > 0:
            prev = plan.folds[i - 1]
            assert fold.train_end == prev.val_end
            assert fold.val_start > prev.val_end


def test_raises_on_insufficient_days():
    days = _days(300)
    with pytest.raises(ValueError):
        build_walk_forward_folds(days, initial_train_days=400, val_block_days=100)


def test_slice_fold_and_holdout_partition_data_without_overlap():
    days = _days(1242)
    plan = build_walk_forward_folds(days, initial_train_days=400, val_block_days=100)
    df = pd.DataFrame({"date": days})

    seen = set()
    for fold in plan.folds:
        train_df, val_df = slice_fold(df, fold, date_col="date")
        assert not (set(train_df["date"]) & set(val_df["date"]))
        seen |= set(val_df["date"])

    holdout_df = slice_holdout(df, plan, date_col="date")
    assert not (seen & set(holdout_df["date"]))
    # every day is accounted for across folds' train+val union and holdout
    all_val_and_holdout = seen | set(holdout_df["date"])
    assert set(days[400:]) == all_val_and_holdout
