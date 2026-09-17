"""
train_final.py
----------------
Trains the FINAL deployable model for a configuration that has already
passed the full promotion gate (Chapter 9 / promotion_gate.py) — this
script does not re-decide whether the config is good, it only produces the
actual deployable artifact for a config that already proved itself.

Fits on every day up to (not including) the untouched holdout block, then
evaluates exactly ONCE on the holdout — the last honest out-of-sample
check before shipping, never touched during the walk-forward validation
that got the config through the gate. Logs the real model artifact (not
just metrics) and registers it in the MLflow Model Registry.

Usage:
    python3 train_final.py --model tcn --seed 0
    python3 train_final.py --model transformer --seed 0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import mlflow
import mlflow.pytorch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from windowing import build_windows, slice_by_date  # noqa: E402
from train_fold import train_and_eval_fold  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline"))
from splits import build_walk_forward_folds  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlflow_config import set_tracking  # noqa: E402


def find_newest_parquet(data_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not candidates:
        raise SystemExit(f"No parquet found in {data_dir}")
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["tcn", "lstm", "transformer"])
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--initial-train-days", type=int, default=400)
    parser.add_argument("--val-block-days", type=int, default=100)
    parser.add_argument("--registered-name", default=None,
                         help="Model Registry name; defaults to banknifty_bigmove_<model>")
    parser.add_argument("--model-kwargs-json", default=None,
                         help="JSON dict of architecture hyperparameters, e.g. a sweep winner's config")
    args = parser.parse_args()

    registered_name = args.registered_name or f"banknifty_bigmove_{args.model}"
    model_kwargs = {}
    if args.model_kwargs_json:
        model_kwargs = json.loads(args.model_kwargs_json)
        if "channels" in model_kwargs:
            model_kwargs["channels"] = tuple(model_kwargs["channels"])

    here = os.path.dirname(os.path.abspath(__file__))
    parquet_path = args.parquet or find_newest_parquet(os.path.join(here, "..", "data", "processed"))
    print(f"Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)

    ds = build_windows(df, lookback=args.lookback)
    trading_days = np.unique(ds.dates)
    plan = build_walk_forward_folds(trading_days, args.initial_train_days, args.val_block_days)

    train_start = plan.folds[0].train_start
    train_end = plan.folds[-1].val_end          # everything through the day before the holdout
    holdout_start, holdout_end = plan.holdout_start, plan.holdout_end

    train_ds = slice_by_date(ds, train_start, train_end)
    holdout_ds = slice_by_date(ds, holdout_start, holdout_end)
    print(f"Final training window: {train_start.date()} -> {train_end.date()} ({len(train_ds.y)} rows)")
    print(f"Holdout (never touched before now): {holdout_start.date()} -> {holdout_end.date()} ({len(holdout_ds.y)} rows)")

    set_tracking()
    with mlflow.start_run(run_name=f"FINAL_{args.model}_seed{args.seed}") as run:
        mlflow.log_params({
            "model": args.model, "lookback": args.lookback, "epochs": args.epochs,
            "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
            "train_start": str(train_start.date()), "train_end": str(train_end.date()),
            "holdout_start": str(holdout_start.date()), "holdout_end": str(holdout_end.date()),
            "dataset_path": os.path.basename(parquet_path), "final_model": True,
            "model_kwargs": json.dumps(model_kwargs),
        })

        fold_metrics, _history, model = train_and_eval_fold(
            model_name=args.model, model_kwargs=model_kwargs,
            X_train=train_ds.X, y_train=train_ds.y,
            X_val=holdout_ds.X, y_val=holdout_ds.y,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            seed=args.seed, device=args.device,
        )
        mlflow.log_metrics({f"holdout_{k}": v for k, v in fold_metrics.as_flat_dict().items()})
        print(f"\nHoldout evaluation (final, honest, never-before-touched):")
        print(f"  roc_auc={fold_metrics.roc_auc:.4f} pr_auc={fold_metrics.pr_auc:.4f} brier={fold_metrics.brier:.4f}")

        mlflow.pytorch.log_model(
            model, artifact_path="model",
            registered_model_name=registered_name,
        )
        print(f"\nLogged + registered as '{registered_name}' (MLflow run {run.info.run_id})")


if __name__ == "__main__":
    main()
