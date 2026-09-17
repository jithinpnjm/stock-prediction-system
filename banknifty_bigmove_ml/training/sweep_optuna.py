"""
sweep_optuna.py
-----------------
Phase 3 real hyperparameter search: architecture-specific search spaces,
evaluated across ALL 7 walk-forward folds every trial (never fewer — this
project doesn't cut that corner even during search), objective = mean PR-AUC
minus its cross-fold std (rewards both strong AND consistent performance,
same spirit as the promotion gate's cross-fold-std check).

Search-phase epochs are deliberately lower than final validation epochs
(--search-epochs, default 8, vs the 15 used everywhere else) to keep each
trial's cost down — the WINNING config from this sweep still needs a full
run_experiment.py pass (15 epochs, all folds) plus 2 seed re-runs and a real
promotion_gate.py check before it's trusted. This script finds candidates,
it does not itself validate one.

Resumable: the Optuna study is backed by a SQLite file on the persistent
volume, so an interrupted sweep picks up where it left off on rerun.

Usage:
    python3 sweep_optuna.py --model tcn --n-trials 20
    python3 sweep_optuna.py --model transformer --n-trials 20 --search-epochs 6
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from windowing import build_windows, slice_by_date  # noqa: E402
from train_fold import train_and_eval_fold  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline"))
from splits import build_walk_forward_folds  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlflow_config import set_tracking  # noqa: E402

import mlflow  # noqa: E402

LOOKBACK_CHOICES = [60, 120, 240]


def find_newest_parquet(data_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not candidates:
        raise SystemExit(f"No parquet found in {data_dir}")
    return candidates[-1]


def suggest_model_kwargs(trial: optuna.Trial, model_name: str) -> dict:
    if model_name == "tcn":
        n_layers = trial.suggest_int("n_layers", 2, 4)
        width = trial.suggest_categorical("width", [16, 32, 64])
        return {
            "channels": tuple([width] * n_layers),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        }
    if model_name == "lstm":
        return {
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        }
    if model_name == "transformer":
        d_model = trial.suggest_categorical("d_model", [16, 32, 64])
        nhead = trial.suggest_categorical("nhead", [2, 4])
        if d_model % nhead != 0:
            raise optuna.TrialPruned(f"d_model={d_model} not divisible by nhead={nhead}")
        return {
            "d_model": d_model, "nhead": nhead,
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        }
    raise ValueError(model_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["tcn", "lstm", "transformer"])
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--search-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--initial-train-days", type=int, default=400)
    parser.add_argument("--val-block-days", type=int, default=100)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage-dir", default="/data/mlops/optuna-studies")
    args = parser.parse_args()

    study_name = args.study_name or f"banknifty_{args.model}_sweep"
    os.makedirs(args.storage_dir, exist_ok=True)
    storage = f"sqlite:///{args.storage_dir}/{study_name}.db"

    here = os.path.dirname(os.path.abspath(__file__))
    parquet_path = args.parquet or find_newest_parquet(os.path.join(here, "..", "data", "processed"))
    print(f"Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # Fold plan computed ONCE from the most lookback-restrictive windowing
    # (the largest lookback drops the most leading rows/days), then reused
    # for every trial regardless of that trial's own lookback choice — this
    # keeps fold boundaries identical across trials for a fair comparison,
    # at the negligible cost of a few days trimmed off the very start of
    # smaller-lookback trials' training window.
    max_lookback = max(LOOKBACK_CHOICES)
    reference_ds = build_windows(df, lookback=max_lookback)
    plan = build_walk_forward_folds(
        np.unique(reference_ds.dates), args.initial_train_days, args.val_block_days
    )
    print(f"{len(plan.folds)} folds (fixed across all trials); holdout untouched: "
          f"{plan.holdout_start.date()} -> {plan.holdout_end.date()}")

    window_cache: dict[int, object] = {}

    def get_windowed(lookback: int):
        if lookback not in window_cache:
            window_cache[lookback] = build_windows(df, lookback=lookback)
        return window_cache[lookback]

    set_tracking()

    def objective(trial: optuna.Trial) -> float:
        lookback = trial.suggest_categorical("lookback", LOOKBACK_CHOICES)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        model_kwargs = suggest_model_kwargs(trial, args.model)

        ds = get_windowed(lookback)
        pr_aucs = []

        with mlflow.start_run(run_name=f"{args.model}_trial{trial.number}", nested=False) as run:
            mlflow.log_params({
                "model": args.model, "lookback": lookback, "lr": lr,
                "search_epochs": args.search_epochs, "study_name": study_name,
                "trial_number": trial.number, **model_kwargs,
            })
            for fold in plan.folds:
                train_ds = slice_by_date(ds, fold.train_start, fold.train_end)
                val_ds = slice_by_date(ds, fold.val_start, fold.val_end)
                with mlflow.start_run(run_name=f"fold{fold.fold_id}", nested=True):
                    mlflow.log_params({"fold_id": fold.fold_id})
                    fm, _hist, _model = train_and_eval_fold(
                        model_name=args.model, model_kwargs=model_kwargs,
                        X_train=train_ds.X, y_train=train_ds.y,
                        X_val=val_ds.X, y_val=val_ds.y,
                        epochs=args.search_epochs, lr=lr, batch_size=args.batch_size,
                        seed=args.seed, device=args.device,
                    )
                    mlflow.log_metrics(fm.as_flat_dict())
                    pr_aucs.append(fm.pr_auc)

            mean_pr_auc = float(np.mean(pr_aucs))
            std_pr_auc = float(np.std(pr_aucs))
            score = mean_pr_auc - std_pr_auc
            mlflow.log_metrics({
                "pr_auc_mean": mean_pr_auc, "pr_auc_std": std_pr_auc, "objective_score": score,
            })
            print(f"[trial {trial.number}] lookback={lookback} lr={lr:.5f} {model_kwargs} "
                  f"-> pr_auc_mean={mean_pr_auc:.4f} std={std_pr_auc:.4f} score={score:.4f}")

        return score

    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize", load_if_exists=True,
    )
    n_done = len(study.trials)
    n_remaining = max(0, args.n_trials - n_done)
    print(f"Study '{study_name}': {n_done} trials already recorded, running {n_remaining} more "
          f"(resumable — rerun this command to continue if interrupted)")

    study.optimize(objective, n_trials=n_remaining)

    print("\n=== Best trial ===")
    print(f"  score: {study.best_value:.4f}")
    print(f"  params: {study.best_trial.params}")


if __name__ == "__main__":
    main()
