"""
build_directional_labeled_dataset.py
---------------------------------------
Directional (triple-barrier) counterpart to build_labeled_dataset.py —
produces long_wins/short_wins columns (see triple_barrier_label.py) instead
of the earlier ±0.3%-any-direction label. Kept as a SEPARATE script/output
file so the original labeled dataset (still used by the already-registered
v1/v2 models) is never touched or invalidated by this addition.

Usage:
    python3 build_directional_labeled_dataset.py
    python3 build_directional_labeled_dataset.py --target-pts 100 --stop-pts 50 --entry-cutoff-minute 900
"""
import argparse
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

from triple_barrier_label import triple_barrier_session

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_HERE)
RAW_SOURCE = os.path.join(
    _PROJECT_DIR, "..", "Banknifty_5 mins trade", "data", "banknifty_spot_1m.csv"
)
RAW_SNAPSHOT_DIR = os.path.join(_PROJECT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(_PROJECT_DIR, "data", "processed")

DEFAULT_TARGET_PTS = 100.0
DEFAULT_STOP_PTS = 50.0
DEFAULT_ENTRY_CUTOFF_MINUTE = 15 * 60  # 15:00 IST — no new entries at/after this


def snapshot_raw(source_path: str) -> str:
    os.makedirs(RAW_SNAPSHOT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    dest = os.path.join(RAW_SNAPSHOT_DIR, f"banknifty_spot_1m_{stamp}.csv")
    shutil.copy2(source_path, dest)
    return dest


def build_directional_labels(df: pd.DataFrame, target_pts: float, stop_pts: float, entry_cutoff_minute: int) -> pd.DataFrame:
    df = df.sort_values("datetime").reset_index(drop=True)
    df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute

    long_wins = np.full(len(df), np.nan)
    short_wins = np.full(len(df), np.nan)
    for _, idx in df.groupby("date").indices.items():
        idx = np.asarray(idx)
        h, l, c, m = (
            df["high"].values[idx], df["low"].values[idx],
            df["close"].values[idx], df["minute_of_day"].values[idx],
        )
        lw, sw = triple_barrier_session(h, l, c, m, target_pts=target_pts, stop_pts=stop_pts,
                                         entry_cutoff_minute=entry_cutoff_minute)
        long_wins[idx] = lw
        short_wins[idx] = sw

    df["long_wins"] = long_wins
    df["short_wins"] = short_wins
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-pts", type=float, default=DEFAULT_TARGET_PTS)
    parser.add_argument("--stop-pts", type=float, default=DEFAULT_STOP_PTS)
    parser.add_argument("--entry-cutoff-minute", type=int, default=DEFAULT_ENTRY_CUTOFF_MINUTE,
                         help="Minutes since midnight; default 900 = 15:00 IST, no new entries at/after this")
    parser.add_argument("--input", default=RAW_SOURCE)
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw source not found: {input_path}")

    print(f"📸 Snapshotting raw source: {input_path}")
    snapshot_path = snapshot_raw(input_path)
    print(f"   -> {snapshot_path}")

    print("📖 Loading + computing directional (triple-barrier) labels per session...")
    raw = pd.read_csv(snapshot_path, parse_dates=["datetime"])
    raw["date"] = raw["datetime"].dt.date
    labeled = build_directional_labels(raw, args.target_pts, args.stop_pts, args.entry_cutoff_minute)

    total_rows = len(labeled)
    valid = labeled.dropna(subset=["long_wins", "short_wins"])
    n_valid = len(valid)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = os.path.join(PROCESSED_DIR, f"banknifty_directional_full_{stamp}.parquet")
    labeled.to_parquet(out_path, index=False)

    long_rate = valid["long_wins"].mean()
    short_rate = valid["short_wins"].mean()
    breakeven = args.stop_pts / (args.target_pts + args.stop_pts)
    both = ((valid["long_wins"] == 1) & (valid["short_wins"] == 1)).mean()
    neither = ((valid["long_wins"] == 0) & (valid["short_wins"] == 0)).mean()

    print("\n📊 Directional label summary")
    print(f"   Target / Stop         : +{args.target_pts} / -{args.stop_pts} pts "
          f"(breakeven win rate: {breakeven:.1%})")
    print(f"   Entry cutoff          : {args.entry_cutoff_minute // 60:02d}:{args.entry_cutoff_minute % 60:02d} IST")
    print(f"   Total rows (kept)     : {total_rows}")
    print(f"   Valid entry candles   : {n_valid}")
    print(f"   Long win rate         : {long_rate:.4f} ({int(valid['long_wins'].sum())} wins)")
    print(f"   Short win rate        : {short_rate:.4f} ({int(valid['short_wins'].sum())} wins)")
    print(f"   Both sides win (rare) : {both:.4f}")
    print(f"   Neither side wins     : {neither:.4f}")
    print(f"\n💾 Saved -> {out_path}")


if __name__ == "__main__":
    main()
