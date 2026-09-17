"""
simulate.py
------------
Replays a registered model over a real historical period (by default, the
untouched holdout — the only period this model has never been evaluated
against before this exact check) using the SAME candle-by-candle inference
path a live deployment would use (agent/inference.py), and reports
trade-relevant statistics at a chosen probability threshold.

Deliberately does NOT invent a P&L number. Simulated rupee profit/loss
depends on real decisions this project hasn't made yet — which option
structure (straddle/strangle vs directional buy), strikes, position size,
slippage, and holding-period rules. Fabricating a P&L here would be exactly
the kind of "looks convincing, isn't validated" result this project's whole
promotion-gate discipline exists to prevent. What IS reported — signal
count, precision (of signals, how many were followed by a real big move),
recall (of real big moves, how many were signaled), and time-to-move — is
everything the model itself can honestly claim, given it predicts
"a big move is coming" without predicting direction.

Usage:
    python3 simulate.py --model tcn --threshold 0.5
    python3 simulate.py --model tcn --threshold 0.5 --start 2026-02-16 --end 2026-09-11
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import BigMoveModel, replay_dataframe  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline"))
from splits import build_walk_forward_folds  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
from windowing import build_windows  # noqa: E402


def find_newest_parquet(data_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not candidates:
        raise SystemExit(f"No parquet found in {data_dir}")
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Registered model name suffix, e.g. 'tcn' -> banknifty_bigmove_tcn")
    parser.add_argument("--stage-or-version", default="latest")
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD; defaults to the dataset's true holdout start")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD; defaults to the dataset's true holdout end")
    parser.add_argument("--initial-train-days", type=int, default=400)
    parser.add_argument("--val-block-days", type=int, default=100)
    parser.add_argument("--list-recent", type=int, default=0,
                         help="Also print the N most recent individual signals (datetime, close, "
                              "probability, outcome) rather than only aggregate stats. Outcome is "
                              "'pending' for signals too recent for the 45-candle horizon to have resolved yet.")
    parser.add_argument("--out-csv", default=None, help="Optionally save ALL individual signals to this CSV path")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    parquet_path = args.parquet or find_newest_parquet(os.path.join(here, "..", "data", "processed"))
    df = pd.read_parquet(parquet_path)

    if args.start and args.end:
        start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    else:
        ds = build_windows(df, lookback=args.lookback)
        plan = build_walk_forward_folds(np.unique(ds.dates), args.initial_train_days, args.val_block_days)
        start, end = plan.holdout_start, plan.holdout_end
        print(f"No --start/--end given — using the dataset's true untouched holdout: {start.date()} -> {end.date()}")

    period_df = df[(pd.to_datetime(df["datetime"]).dt.date >= start.date()) &
                    (pd.to_datetime(df["datetime"]).dt.date <= end.date())].copy()
    print(f"Replaying {len(period_df)} candles from {start.date()} to {end.date()}")

    registered_name = f"banknifty_bigmove_{args.model}"
    model = BigMoveModel(registered_name, args.stage_or_version, lookback=args.lookback)

    preds = replay_dataframe(model, period_df)
    print(f"Produced {len(preds)} predictions (first {args.lookback} candles of the period have no prediction — insufficient lookback)")

    # Ground truth: did a >=0.3% move actually happen in the next 45 candles? Reuse the SAME
    # label logic the model was trained against, applied fresh to this exact period for honesty.
    sys.path.insert(0, os.path.join(here, "..", "data_pipeline"))
    from label_logic import label_session  # noqa: E402

    period_sorted = period_df.sort_values("datetime").reset_index(drop=True)
    labels = np.full(len(period_sorted), np.nan)
    for _, idx in period_sorted.groupby(period_sorted["datetime"].dt.date).indices.items():
        idx = np.asarray(idx)
        labels[idx] = label_session(
            period_sorted["high"].values[idx], period_sorted["low"].values[idx],
            period_sorted["close"].values[idx], horizon=45, pct=0.003,
        )
    period_sorted["label"] = labels

    # Keep ALL predictions (including undefined-outcome rows) for the signal listing/CSV —
    # only the aggregate precision/recall calc needs to exclude rows whose outcome isn't
    # knowable yet (the last ~45 candles of the period, which haven't had their forward
    # window resolve). Dropping those rows entirely would hide the most recent signal
    # from a "what's the latest prediction" view, which is exactly what's wanted here.
    merged = preds.merge(period_sorted[["datetime", "label"]], on="datetime", how="left")
    merged["signal"] = merged["probability"] >= args.threshold
    merged["outcome"] = merged["label"].map({1.0: "big_move", 0.0: "no_move"}).fillna("pending")

    resolved = merged.dropna(subset=["label"])
    n_signals = int(resolved["signal"].sum())
    n_actual_moves = int(resolved["label"].sum())
    n_correct_signals = int((resolved["signal"] & (resolved["label"] == 1)).sum())
    n_caught_moves = n_correct_signals  # same count, viewed from the other side

    precision = n_correct_signals / n_signals if n_signals else float("nan")
    recall = n_caught_moves / n_actual_moves if n_actual_moves else float("nan")

    print("\n=== Simulation report (NOT a P&L — see module docstring for why) ===")
    print(f"  Period               : {start.date()} -> {end.date()}")
    print(f"  Threshold            : {args.threshold}")
    print(f"  Predictions made     : {len(merged)} ({len(merged) - len(resolved)} too recent to have a resolved outcome yet)")
    print(f"  Signals (P>=thresh)  : {n_signals} (of resolved predictions)")
    print(f"  Actual big moves     : {n_actual_moves}")
    print(f"  Signals that were right (precision): {precision:.3f}" if n_signals else "  No signals at this threshold")
    print(f"  Real moves caught (recall)         : {recall:.3f}" if n_actual_moves else "  No real moves in this period")
    print("\nNote: the model predicts THAT a big move is coming, not its direction.")
    print("Translating a signal into a real trade (straddle vs directional, strikes, sizing,")
    print("slippage, holding period) is a separate design decision this report deliberately")
    print("does not make for you.")

    if args.list_recent:
        recent_signals = merged[merged["signal"]].sort_values("datetime", ascending=False).head(args.list_recent)
        print(f"\n=== {len(recent_signals)} most recent signals (P>={args.threshold}) ===")
        for row in recent_signals.itertuples(index=False):
            print(f"  {row.datetime} | close={row.close:.1f} | P={row.probability:.3f} | outcome={row.outcome}")

    if args.out_csv:
        merged.sort_values("datetime").to_csv(args.out_csv, index=False)
        print(f"\nSaved all {len(merged)} predictions -> {args.out_csv}")


if __name__ == "__main__":
    main()
