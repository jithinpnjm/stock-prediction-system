"""
run_experiment.py
-------------------
Phase 2/3 entrypoint: trains one hyperparameter config across every
walk-forward fold, logs everything to MLflow (nested runs: parent = config,
child = fold — see project plan), and prints the cross-fold summary the
promotion gate later consumes.

Never trusts a single run: every fold is a separate logged child run, and
the parent aggregates mean/std/MIN (not just mean) across them.

Usage:
    python3 run_experiment.py --model tcn --lookback 120 --epochs 5 --seed 0
    python3 run_experiment.py --model baseline_logreg --lookback 120   # reference run for the promotion gate
    python3 run_experiment.py --model tcn --smoke                      # 1 fold, 2 epochs, tiny subset — plumbing check only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline"))
from splits import build_walk_forward_folds, describe  # noqa: E402

from windowing import build_windows, slice_by_date  # noqa: E402
from train_fold import train_and_eval_fold  # noqa: E402
from baseline import naive_predict, logreg_predict  # noqa: E402
from metrics import compute_fold_metrics  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlflow_config import set_tracking  # noqa: E402

import mlflow  # noqa: E402


def find_newest_parquet(data_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not candidates:
        raise SystemExit(f"No parquet found in {data_dir} — run data_pipeline/build_labeled_dataset.py first.")
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                         choices=["tcn", "lstm", "transformer", "baseline_naive", "baseline_logreg"])
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-kwargs-json", default=None,
                         help='JSON dict of architecture hyperparameters, e.g. from a sweep winner: '
                              '\'{"channels": [16,16,16], "kernel_size": 3, "dropout": 0.16}\'')
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    parser.add_argument("--initial-train-days", type=int, default=400)
    parser.add_argument("--val-block-days", type=int, default=100)
    parser.add_argument("--max-folds", type=int, default=None, help="Cap number of folds run (smoke tests)")
    parser.add_argument("--smoke", action="store_true", help="Tiny/fast run to check the plumbing, not a real result")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.max_folds = 1

    model_kwargs = {}
    if args.model_kwargs_json:
        model_kwargs = json.loads(args.model_kwargs_json)
        if "channels" in model_kwargs:
            model_kwargs["channels"] = tuple(model_kwargs["channels"])  # TCN expects a tuple, JSON gives a list

    here = os.path.dirname(os.path.abspath(__file__))
    parquet_path = args.parquet or find_newest_parquet(os.path.join(here, "..", "data", "processed"))
    print(f"Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print(f"Building windows (lookback={args.lookback})...")
    t0 = time.time()
    ds = build_windows(df, lookback=args.lookback)
    print(f"  -> {len(ds.y)} labeled windows in {time.time() - t0:.1f}s")

    trading_days = np.unique(ds.dates)
    plan = build_walk_forward_folds(
        trading_days, initial_train_days=args.initial_train_days, val_block_days=args.val_block_days
    )
    print(describe(plan))

    folds = plan.folds[: args.max_folds] if args.max_folds else plan.folds

    set_tracking()
    run_name = f"{args.model}_lb{args.lookback}_seed{args.seed}"
    fold_results = []

    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.log_params({
            "model": args.model, "lookback": args.lookback, "epochs": args.epochs,
            "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
            "model_kwargs": json.dumps(model_kwargs),
            "device": args.device, "n_folds_total": len(plan.folds), "n_folds_run": len(folds),
            "initial_train_days": args.initial_train_days, "val_block_days": args.val_block_days,
            "dataset_path": os.path.basename(parquet_path), "smoke_test": args.smoke,
        })

        for fold in folds:
            train_start = pd.Timestamp(fold.train_start)
            train_end = pd.Timestamp(fold.train_end)
            val_start = pd.Timestamp(fold.val_start)
            val_end = pd.Timestamp(fold.val_end)

            train_ds = slice_by_date(ds, train_start, train_end)
            val_ds = slice_by_date(ds, val_start, val_end)
            print(f"\n[fold {fold.fold_id}] train {len(train_ds.y)} rows, val {len(val_ds.y)} rows")

            with mlflow.start_run(run_name=f"fold{fold.fold_id}", nested=True):
                mlflow.log_params({
                    "fold_id": fold.fold_id,
                    "train_start": str(fold.train_start.date()), "train_end": str(fold.train_end.date()),
                    "val_start": str(fold.val_start.date()), "val_end": str(fold.val_end.date()),
                })

                if args.model == "baseline_naive":
                    probs = naive_predict(train_ds.y, len(val_ds.y))
                    fm = compute_fold_metrics(val_ds.y, probs)
                elif args.model == "baseline_logreg":
                    probs = logreg_predict(train_ds.X, train_ds.y, val_ds.X)
                    fm = compute_fold_metrics(val_ds.y, probs)
                else:
                    fm, _history, _model = train_and_eval_fold(
                        model_name=args.model, model_kwargs=model_kwargs,
                        X_train=train_ds.X, y_train=train_ds.y,
                        X_val=val_ds.X, y_val=val_ds.y,
                        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                        seed=args.seed, device=args.device,
                    )

                mlflow.log_metrics(fm.as_flat_dict())
                print(f"  roc_auc={fm.roc_auc:.4f} pr_auc={fm.pr_auc:.4f} brier={fm.brier:.4f}")
                fold_results.append((fold.fold_id, fm))

        # Cross-fold aggregate — mean/std AND min, so one bad fold isn't averaged away
        metric_names = ["roc_auc", "pr_auc", "brier"]
        agg = {}
        for name in metric_names:
            values = [getattr(fm, name) for _, fm in fold_results if not np.isnan(getattr(fm, name))]
            if values:
                agg[f"{name}_mean"] = float(np.mean(values))
                agg[f"{name}_std"] = float(np.std(values))
                agg[f"{name}_min"] = float(np.min(values))
        mlflow.log_metrics(agg)

        summary_df = pd.DataFrame([
            {"fold_id": fid, **fm.as_flat_dict()} for fid, fm in fold_results
        ])
        summary_path = "/tmp/fold_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        mlflow.log_artifact(summary_path)

        print("\n=== Cross-fold summary ===")
        print(summary_df.to_string(index=False))
        print("\n=== Aggregate ===")
        for k, v in agg.items():
            print(f"  {k}: {v:.4f}")
        print(f"\nMLflow parent run_id: {parent_run.info.run_id}")


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
