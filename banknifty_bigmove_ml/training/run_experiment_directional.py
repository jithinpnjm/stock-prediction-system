"""
run_experiment_directional.py
--------------------------------
Directional (triple-barrier) counterpart to run_experiment.py — the REAL
training target locked 2026-09-13: two heads per model (P(long_wins),
P(short_wins)) against build_directional_labeled_dataset.py's output,
evaluated both per-head (ROC-AUC/PR-AUC, same promotion-gate-compatible
shape as before) AND at the strategy level (actual win rate / expected
points-per-trade if you took whichever direction the model favored above
threshold) — see metrics.compute_directional_strategy_metrics.

Kept as a separate script from run_experiment.py rather than branching
inside it, so the original (already-registered v1/v2) single-target
pipeline is never touched by this addition.

Usage:
    python3 run_experiment_directional.py --model tcn --lookback 120 --epochs 5 --seed 0
    python3 run_experiment_directional.py --model tcn --smoke   # 1 fold, 2 epochs — plumbing check only
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

from windowing import build_windows_directional, slice_directional_by_date, N_FEATURES_DIRECTIONAL  # noqa: E402
from train_fold import train_and_eval_fold_directional  # noqa: E402
from baseline import naive_predict, logreg_predict  # noqa: E402
from metrics import compute_fold_metrics, compute_directional_strategy_metrics  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlflow_config import set_tracking  # noqa: E402

import mlflow  # noqa: E402


def find_newest_directional_parquet(data_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(data_dir, "banknifty_directional_full_*.parquet")))
    if not candidates:
        raise SystemExit(f"No directional parquet found in {data_dir} — run "
                          f"data_pipeline/build_directional_labeled_dataset.py first.")
    return candidates[-1]


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


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
    parser.add_argument("--strategy-threshold", type=float, default=0.5,
                         help="Confidence threshold for taking a trade in either direction")
    parser.add_argument("--model-kwargs-json", default=None)
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    parser.add_argument("--initial-train-days", type=int, default=400)
    parser.add_argument("--val-block-days", type=int, default=100)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.max_folds = 1

    model_kwargs = {}
    if args.model_kwargs_json:
        model_kwargs = json.loads(args.model_kwargs_json)
        if "channels" in model_kwargs:
            model_kwargs["channels"] = tuple(model_kwargs["channels"])
    # directional windowing adds prev-day context channels (13, not the base 10) —
    # models.py's classes default n_features to the OLD N_FEATURES at import time,
    # so this must be passed explicitly or every architecture mismatches its input.
    model_kwargs.setdefault("n_features", N_FEATURES_DIRECTIONAL)

    here = os.path.dirname(os.path.abspath(__file__))
    parquet_path = args.parquet or find_newest_directional_parquet(os.path.join(here, "..", "data", "processed"))
    print(f"Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if args.smoke:
        # Windowing the FULL ~460k-row history (13 channels incl. the new prev-day
        # context) uses several GB of RAM — fine on a real training box, but this
        # OOM-killed a small CPU VM during a plumbing check. A smoke test only
        # needs to prove the wiring works, so shrink the input BEFORE windowing,
        # not just cap fold count afterward (the OOM happens before folds exist).
        recent_days = sorted(df["date"].unique())[-60:]
        df = df[df["date"].isin(recent_days)].reset_index(drop=True)
        args.initial_train_days = min(args.initial_train_days, 30)
        args.val_block_days = min(args.val_block_days, 10)
        print(f"  --smoke: truncated to the most recent {len(recent_days)} days "
              f"({len(df)} rows) before windowing, to keep memory small")

    print(f"Building directional windows (lookback={args.lookback})...")
    t0 = time.time()
    ds = build_windows_directional(df, lookback=args.lookback)
    print(f"  -> {len(ds.y)} labeled windows in {time.time() - t0:.1f}s")

    trading_days = np.unique(ds.dates)
    plan = build_walk_forward_folds(trading_days, args.initial_train_days, args.val_block_days)
    print(describe(plan))
    folds = plan.folds[: args.max_folds] if args.max_folds else plan.folds

    set_tracking()
    run_name = f"dir_{args.model}_lb{args.lookback}_seed{args.seed}"
    fold_results = []

    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.log_params({
            "model": args.model, "lookback": args.lookback, "epochs": args.epochs,
            "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
            "strategy_threshold": args.strategy_threshold, "model_kwargs": json.dumps(model_kwargs),
            "device": args.device, "n_folds_total": len(plan.folds), "n_folds_run": len(folds),
            "initial_train_days": args.initial_train_days, "val_block_days": args.val_block_days,
            "dataset_path": os.path.basename(parquet_path), "smoke_test": args.smoke,
            "directional": True,
        })

        for fold in folds:
            train_ds = slice_directional_by_date(ds, fold.train_start, fold.train_end)
            val_ds = slice_directional_by_date(ds, fold.val_start, fold.val_end)
            print(f"\n[fold {fold.fold_id}] train {len(train_ds.y)} rows, val {len(val_ds.y)} rows")

            with mlflow.start_run(run_name=f"fold{fold.fold_id}", nested=True):
                mlflow.log_params({
                    "fold_id": fold.fold_id,
                    "train_start": str(fold.train_start.date()), "train_end": str(fold.train_end.date()),
                    "val_start": str(fold.val_start.date()), "val_end": str(fold.val_end.date()),
                })

                if args.model == "baseline_naive":
                    p_long = naive_predict(train_ds.y[:, 0], len(val_ds.y))
                    p_short = naive_predict(train_ds.y[:, 1], len(val_ds.y))
                elif args.model == "baseline_logreg":
                    p_long = logreg_predict(train_ds.X, train_ds.y[:, 0], val_ds.X)
                    p_short = logreg_predict(train_ds.X, train_ds.y[:, 1], val_ds.X)
                else:
                    long_m, short_m, strat_m, _hist, _model = train_and_eval_fold_directional(
                        model_name=args.model, model_kwargs=model_kwargs,
                        X_train=train_ds.X, y_train=train_ds.y,
                        X_val=val_ds.X, y_val=val_ds.y,
                        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                        seed=args.seed, device=args.device, strategy_threshold=args.strategy_threshold,
                    )
                    p_long = p_short = None  # already scored via long_m/short_m/strat_m below

                if p_long is not None:  # baseline path — compute metrics the same way train_and_eval_fold_directional does
                    long_m = compute_fold_metrics(val_ds.y[:, 0], p_long)
                    short_m = compute_fold_metrics(val_ds.y[:, 1], p_short)
                    strat_m = compute_directional_strategy_metrics(
                        p_long, p_short, val_ds.y[:, 0], val_ds.y[:, 1], threshold=args.strategy_threshold,
                    )

                mlflow.log_metrics({f"long_{k}": v for k, v in long_m.as_flat_dict().items()})
                mlflow.log_metrics({f"short_{k}": v for k, v in short_m.as_flat_dict().items()})
                mlflow.log_metrics({
                    "strategy_n_trades": strat_m.n_trades, "strategy_n_long": strat_m.n_long,
                    "strategy_n_short": strat_m.n_short,
                    "strategy_win_rate": strat_m.win_rate if not np.isnan(strat_m.win_rate) else -1.0,
                    "strategy_long_win_rate": strat_m.long_win_rate if not np.isnan(strat_m.long_win_rate) else -1.0,
                    "strategy_short_win_rate": strat_m.short_win_rate if not np.isnan(strat_m.short_win_rate) else -1.0,
                    "strategy_expected_points": strat_m.expected_points_per_trade if not np.isnan(strat_m.expected_points_per_trade) else -9999.0,
                })
                print(f"  long: roc_auc={long_m.roc_auc:.4f} pr_auc={long_m.pr_auc:.4f} | "
                      f"short: roc_auc={short_m.roc_auc:.4f} pr_auc={short_m.pr_auc:.4f}")
                print(f"  strategy: {strat_m.n_trades} trades ({strat_m.n_long} long/{strat_m.n_short} short), "
                      f"win_rate={strat_m.win_rate:.3f}, expected_pts/trade={strat_m.expected_points_per_trade:.2f}")
                fold_results.append((fold.fold_id, long_m, short_m, strat_m))

        # Cross-fold aggregate — mean/std AND min for the per-head metrics, plus a
        # pooled (not averaged) strategy win rate, since trade COUNT varies by fold
        # and a simple mean-of-means would misweight folds with few trades.
        agg = {}
        for label, idx in [("long", 1), ("short", 2)]:
            for name in ["roc_auc", "pr_auc"]:
                values = [getattr(fr[idx], name) for fr in fold_results if not np.isnan(getattr(fr[idx], name))]
                if values:
                    agg[f"{label}_{name}_mean"] = float(np.mean(values))
                    agg[f"{label}_{name}_std"] = float(np.std(values))
                    agg[f"{label}_{name}_min"] = float(np.min(values))

        total_trades = sum(fr[3].n_trades for fr in fold_results)
        total_wins = sum(fr[3].n_trades * fr[3].win_rate for fr in fold_results if not np.isnan(fr[3].win_rate))
        pooled_win_rate = total_wins / total_trades if total_trades else float("nan")
        agg["pooled_win_rate"] = pooled_win_rate
        agg["total_trades"] = total_trades
        agg["pooled_expected_points"] = (
            pooled_win_rate * 100.0 - (1 - pooled_win_rate) * 50.0 if total_trades else float("nan")
        )
        mlflow.log_metrics({k: v for k, v in agg.items() if not (isinstance(v, float) and np.isnan(v))})

        summary_df = pd.DataFrame([
            {"fold_id": fid, "long_roc_auc": lm.roc_auc, "long_pr_auc": lm.pr_auc,
             "short_roc_auc": sm.roc_auc, "short_pr_auc": sm.pr_auc,
             "n_trades": st.n_trades, "win_rate": st.win_rate, "expected_pts": st.expected_points_per_trade}
            for fid, lm, sm, st in fold_results
        ])
        summary_path = "/tmp/directional_fold_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        mlflow.log_artifact(summary_path)

        print("\n=== Cross-fold summary ===")
        print(summary_df.to_string(index=False))
        print("\n=== Aggregate (pooled across all trades taken) ===")
        print(f"  total_trades: {total_trades}")
        print(f"  pooled_win_rate: {pooled_win_rate:.4f}" if not np.isnan(pooled_win_rate) else "  pooled_win_rate: n/a (no trades)")
        print(f"  pooled_expected_points_per_trade: {agg.get('pooled_expected_points', float('nan')):.2f}")
        print(f"\nMLflow parent run_id: {parent_run.info.run_id}")


if __name__ == "__main__":
    main()
