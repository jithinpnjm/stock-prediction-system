"""
build_labeled_dataset.py
--------------------------
Phase 1 of the BankNifty "big move" ML pipeline (see
/Users/jithinpjoseph/.claude/plans/unified-wiggling-wirth.md).

Turns the raw, continuously-updated
`Banknifty_5 mins trade/data/banknifty_spot_1m.csv` into a versioned,
labeled dataset:

  1. Stamps an immutable timestamped copy of the raw CSV into
     `data/raw/` — the source file keeps changing as the downloader runs
     again, so every dataset version must freeze its own input rather than
     silently drifting.
  2. Computes the locked label (see label_logic.py) per trading session
     (calendar day), never leaking across day boundaries or beyond the end
     of the available data.
  3. Writes the labeled dataset to `data/processed/` as parquet and prints
     summary stats (row counts, positive rate, date range) that should be
     eyeballed once and then tracked in MLflow going forward, not trusted
     blindly again.

Usage:
    python3 build_labeled_dataset.py
    python3 build_labeled_dataset.py --horizon 45 --pct 0.003
"""
from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

from label_logic import label_session

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_HERE)
RAW_SOURCE = os.path.join(
    _PROJECT_DIR, "..", "Banknifty_5 mins trade", "data", "banknifty_spot_1m.csv"
)
RAW_SNAPSHOT_DIR = os.path.join(_PROJECT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(_PROJECT_DIR, "data", "processed")

DEFAULT_HORIZON = 45
DEFAULT_PCT = 0.003


def snapshot_raw(source_path: str) -> str:
    os.makedirs(RAW_SNAPSHOT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    dest = os.path.join(RAW_SNAPSHOT_DIR, f"banknifty_spot_1m_{stamp}.csv")
    shutil.copy2(source_path, dest)
    return dest


def build_labels(df: pd.DataFrame, horizon: int, pct: float) -> pd.DataFrame:
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date

    labels = np.empty(len(df))
    labels[:] = np.nan
    for _, idx in df.groupby("date").indices.items():
        idx = np.asarray(idx)
        high = df["high"].values[idx]
        low = df["low"].values[idx]
        close = df["close"].values[idx]
        labels[idx] = label_session(high, low, close, horizon=horizon, pct=pct)

    df["label"] = labels
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                         help="Forward candles to look for the touch (default 45)")
    parser.add_argument("--pct", type=float, default=DEFAULT_PCT,
                         help="Move threshold as a fraction, e.g. 0.003 for 0.3%%")
    parser.add_argument("--input", default=RAW_SOURCE, help="Raw spot 1-min CSV path")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw source not found: {input_path}")

    print(f"📸 Snapshotting raw source: {input_path}")
    snapshot_path = snapshot_raw(input_path)
    print(f"   -> {snapshot_path}")

    print("📖 Loading + labeling (per trading session, no cross-day leakage)...")
    raw = pd.read_csv(snapshot_path, parse_dates=["datetime"])
    labeled = build_labels(raw, horizon=args.horizon, pct=args.pct)

    total_rows = len(labeled)
    n_labeled = labeled["label"].notna().sum()
    dropped = total_rows - n_labeled

    # Keep EVERY row, including the ones with label=NaN (a day's last `horizon`
    # candles). Those rows are still needed as input-window context for a
    # later day's early candles — a model predicting on 09:15 needs lookback
    # history that includes the prior day's last candles, which never get a
    # label themselves. Dropping them here would silently break window
    # continuity across every single session boundary. Training code is
    # responsible for treating label.isna() rows as "context only, not a
    # target" — never as a training/eval sample.
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = os.path.join(PROCESSED_DIR, f"banknifty_bigmove_full_{stamp}.parquet")
    labeled.to_parquet(out_path, index=False)

    pos_rate = labeled["label"].mean()  # mean() skips NaN automatically
    print("\n📊 Label summary")
    print(f"   Horizon              : {args.horizon} candles, threshold ±{args.pct * 100:.2f}%")
    print(f"   Total rows (kept)    : {total_rows}")
    print(f"   Rows with a label    : {n_labeled}")
    print(f"   Rows w/o label (ctx) : {dropped} (session tail — context-only, never a target)")
    print(f"   Date range           : {labeled['datetime'].min()} -> {labeled['datetime'].max()}")
    print(f"   Positive rate        : {pos_rate:.4f} (of labeled rows, {int(labeled['label'].sum())} positives)")
    print(f"\n💾 Saved -> {out_path}")
    print("   (this should be `dvc add`-ed once DVC is wired up — see infra/README.md)")


if __name__ == "__main__":
    main()
