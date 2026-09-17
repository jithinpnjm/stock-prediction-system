"""
run_experiment_mtf.py
------------------------
Multi-timeframe counterpart to run_experiment.py: trains a MultiTimeframeModel
(one encoder branch per timeframe, e.g. 1-min + 5-min, fused before a shared
head) across every walk-forward fold, with the same MLflow nested-run
logging (parent = config, child = fold) and never-trust-one-run discipline.

Usage:
    python3 run_experiment_mtf.py --branch-1min tcn --branch-5min lstm --smoke
    python3 run_experiment_mtf.py --branch-1min tcn --branch-5min tcn --lookback-1min 120 --lookback-5min 24 --epochs 5
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from windowing import build_multi_timeframe_windows, slice_mtf_by_date  # noqa: E402
from train_fold import train_and_eval_fold  # noqa: E402
from metrics import compute_fold_metrics  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline"))
from splits import build_walk_forward_folds, describe  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlflow_config import set_tracking  # noqa: E402

import mlflow  # noqa: E402


def find_newest_parquet(data_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not candidates:
        raise SystemExit(f"No parquet found in {data_dir} — run data_pipeline/build_labeled_dataset.py first.")
    return candidates[-1]


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--branch-1min", required=True, choices=["tcn", "lstm", "transformer"])
    parser.add_argument("--branch-5min", required=True, choices=["tcn", "lstm", "transformer"])
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--lookback-1min", type=int, default=120)
    parser.add_argument("--lookback-5min", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    parser.add_argument("--initial-train-days", type=int, default=400)
    parser.add_argument("--val-block-days", type=int, default=100)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.max_folds = 1

    here = os.path.dirname(os.path.abspath(__file__))
    parquet_path = args.parquet or find_newest_parquet(os.path.join(here, "..", "data", "processed"))
    print(f"Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print(f"Building multi-timeframe windows (1min lookback={args.lookback_1min}, 5min lookback={args.lookback_5min})...")
    t0 = time.time()
    ds = build_multi_timeframe_windows(
        df, base_lookback=args.lookback_1min, htf_specs={"5min": args.lookback_5min}
    )
    print(f"  -> {len(ds.y)} labeled windows in {time.time() - t0:.1f}s")

    trading_days = np.unique(ds.dates)
    plan = build_walk_forward_folds(
        trading_days, initial_train_days=args.initial_train_days, val_block_days=args.val_block_days
    )
    print(describe(plan))
    folds = plan.folds[: args.max_folds] if args.max_folds else plan.folds

    set_tracking()
    run_name = f"mtf_{args.branch_1min}1m_{args.branch_5min}5m_seed{args.seed}"
    fold_results = []

    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.log_params({
            "branch_1min": args.branch_1min, "branch_5min": args.branch_5min,
            "lookback_1min": args.lookback_1min, "lookback_5min": args.lookback_5min,
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
            "device": args.device, "n_folds_total": len(plan.folds), "n_folds_run": len(folds),
            "initial_train_days": args.initial_train_days, "val_block_days": args.val_block_days,
            "dataset_path": os.path.basename(parquet_path), "smoke_test": args.smoke,
        })

        for fold in folds:
            train_ds = slice_mtf_by_date(ds, fold.train_start, fold.train_end)
            val_ds = slice_mtf_by_date(ds, fold.val_start, fold.val_end)
            print(f"\n[fold {fold.fold_id}] train {len(train_ds.y)} rows, val {len(val_ds.y)} rows")

            with mlflow.start_run(run_name=f"fold{fold.fold_id}", nested=True):
                mlflow.log_params({
                    "fold_id": fold.fold_id,
                    "train_start": str(fold.train_start.date()), "train_end": str(fold.train_end.date()),
                    "val_start": str(fold.val_start.date()), "val_end": str(fold.val_end.date()),
                })

                branch_specs = {
                    "1min": (args.branch_1min, {}),
                    "5min": (args.branch_5min, {}),
                }
                fm, _history, _model = train_and_eval_fold(
                    model_name="mtf", model_kwargs={"branch_specs": branch_specs},
                    X_train=train_ds.X_by_tf, y_train=train_ds.y,
                    X_val=val_ds.X_by_tf, y_val=val_ds.y,
                    epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                    seed=args.seed, device=args.device,
                )

                mlflow.log_metrics(fm.as_flat_dict())
                print(f"  roc_auc={fm.roc_auc:.4f} pr_auc={fm.pr_auc:.4f} brier={fm.brier:.4f}")
                fold_results.append((fold.fold_id, fm))

        metric_names = ["roc_auc", "pr_auc", "brier"]
        agg = {}
        for name in metric_names:
            values = [getattr(fm, name) for _, fm in fold_results if not np.isnan(getattr(fm, name))]
            if values:
                agg[f"{name}_mean"] = float(np.mean(values))
                agg[f"{name}_std"] = float(np.std(values))
                agg[f"{name}_min"] = float(np.min(values))
        mlflow.log_metrics(agg)

        summary_df = pd.DataFrame([{"fold_id": fid, **fm.as_flat_dict()} for fid, fm in fold_results])
        summary_path = "/tmp/mtf_fold_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        mlflow.log_artifact(summary_path)

        print("\n=== Cross-fold summary ===")
        print(summary_df.to_string(index=False))
        print("\n=== Aggregate ===")
        for k, v in agg.items():
            print(f"  {k}: {v:.4f}")
        print(f"\nMLflow parent run_id: {parent_run.info.run_id}")


if __name__ == "__main__":
    main()
