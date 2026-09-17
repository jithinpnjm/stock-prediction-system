"""
splits.py
----------
Chronological, expanding-window walk-forward fold definitions for the
labeled BankNifty dataset. Never randomly shuffled — folds are defined over
calendar trading days, and this module is the single source of truth every
training run must import, so all experiments are compared on identical
fold boundaries.

Scheme (see project plan): expanding train window, non-overlapping
validation blocks, one final block reserved as an untouched holdout that
never participates in fold selection/hyperparameter tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp      # inclusive
    val_start: pd.Timestamp
    val_end: pd.Timestamp        # inclusive


@dataclass(frozen=True)
class SplitPlan:
    folds: List[Fold]
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp    # inclusive


def build_walk_forward_folds(
    trading_days: pd.Series | List,
    initial_train_days: int = 400,
    val_block_days: int = 100,
) -> SplitPlan:
    """
    trading_days: sorted, deduplicated calendar dates present in the dataset.

    Produces expanding-window folds:
      fold 1: train[0:initial_train_days]              val[initial_train_days : +val_block_days]
      fold 2: train[0:initial_train_days+val_block_days] val[next block]
      ...
    The final val_block_days-sized block is withheld as `holdout` and is
    NOT included in `folds` — it must never be touched during model
    selection/hyperparameter tuning.
    """
    days = sorted(pd.to_datetime(pd.Series(list(trading_days)).unique()))
    n = len(days)

    if n <= initial_train_days + val_block_days:
        raise ValueError(
            f"Not enough trading days ({n}) for initial_train_days="
            f"{initial_train_days} + val_block_days={val_block_days}"
        )

    folds = []
    train_end_idx = initial_train_days
    fold_id = 1
    while train_end_idx + val_block_days <= n:
        val_start_idx = train_end_idx
        val_end_idx = train_end_idx + val_block_days
        is_last_block = (val_end_idx + val_block_days > n)
        if is_last_block:
            # reserve this final block as the untouched holdout instead of a fold
            break
        folds.append(Fold(
            fold_id=fold_id,
            train_start=days[0],
            train_end=days[val_start_idx - 1],
            val_start=days[val_start_idx],
            val_end=days[val_end_idx - 1],
        ))
        fold_id += 1
        train_end_idx = val_end_idx

    # Holdout absorbs everything from here to the end of the dataset — never
    # capped at exactly val_block_days, so a remainder that doesn't divide
    # evenly (e.g. the newest few weeks) is never silently dropped.
    holdout_start_idx = train_end_idx
    holdout_end_idx = n - 1

    if not folds:
        raise ValueError(
            "Split produced zero folds before the holdout block — "
            "reduce initial_train_days/val_block_days or check trading_days input."
        )

    return SplitPlan(
        folds=folds,
        holdout_start=days[holdout_start_idx],
        holdout_end=days[holdout_end_idx],
    )


def slice_fold(df: pd.DataFrame, fold: Fold, date_col: str = "date"):
    """Return (train_df, val_df) for one fold. df[date_col] must be date-like."""
    dates = pd.to_datetime(df[date_col])
    train_mask = (dates >= fold.train_start) & (dates <= fold.train_end)
    val_mask = (dates >= fold.val_start) & (dates <= fold.val_end)
    return df[train_mask], df[val_mask]


def slice_holdout(df: pd.DataFrame, plan: SplitPlan, date_col: str = "date"):
    dates = pd.to_datetime(df[date_col])
    mask = (dates >= plan.holdout_start) & (dates <= plan.holdout_end)
    return df[mask]


def describe(plan: SplitPlan) -> str:
    lines = [f"{len(plan.folds)} fold(s):"]
    for f in plan.folds:
        lines.append(
            f"  fold {f.fold_id}: train {f.train_start.date()} -> {f.train_end.date()} "
            f"| val {f.val_start.date()} -> {f.val_end.date()}"
        )
    lines.append(f"  holdout (untouched): {plan.holdout_start.date()} -> {plan.holdout_end.date()}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import glob
    import os

    parser = argparse.ArgumentParser(description="Print the walk-forward fold plan for a labeled dataset")
    parser.add_argument("--parquet", default=None, help="Path to a labeled parquet; defaults to the newest in data/processed/")
    parser.add_argument("--initial-train-days", type=int, default=400)
    parser.add_argument("--val-block-days", type=int, default=100)
    args = parser.parse_args()

    parquet_path = args.parquet
    if parquet_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = sorted(glob.glob(os.path.join(here, "..", "data", "processed", "*.parquet")))
        if not candidates:
            raise SystemExit("No labeled parquet found in data/processed/ — run build_labeled_dataset.py first.")
        parquet_path = candidates[-1]

    print(f"Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    plan = build_walk_forward_folds(
        df["date"].unique(),
        initial_train_days=args.initial_train_days,
        val_block_days=args.val_block_days,
    )
    print(describe(plan))
