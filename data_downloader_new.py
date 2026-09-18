"""Append-only Fyers downloader for Bank Nifty 1-minute source data.

Credentials are read from secrets/fyers_auth.json or FYERS_* environment
variables. The real credential file is intentionally ignored by Git.

Downloaded windows are written as immutable Parquet files below
data/raw/fyers/. Re-running the script only requests trading days that are
missing or incomplete according to the configured NSE calendar.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from src.data.calendar import NSECalendar
from src.data.fyers import FyersConfig, create_fyers_client, fetch_window

RAW_DIR = Path("data/raw/fyers")
STATE_FILE = RAW_DIR / "no_data_days.json"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=date.fromisoformat)
    p.add_argument("--end", type=date.fromisoformat)
    p.add_argument("--years-back", type=int, default=5)
    p.add_argument("--auth-file", default=None)
    p.add_argument("--calendar", default="configs/data/nse_holidays.yaml")
    return p.parse_args()


def _known_no_data() -> set[date]:
    if not STATE_FILE.exists():
        return set()
    return {date.fromisoformat(x) for x in json.loads(STATE_FILE.read_text())}


def _save_no_data(days: set[date]):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(sorted(x.isoformat() for x in days), indent=2) + "\n"
    )


def _existing_counts() -> dict[date, int]:
    if not RAW_DIR.exists():
        return {}
    files=list(RAW_DIR.glob("*.parquet"))
    if not files:
        return {}
    df=pl.scan_parquet([str(x) for x in files]).select("timestamp").collect()
    if df.is_empty():
        return {}
    return {
        row[0]: int(row[1])
        for row in df.group_by(pl.col("timestamp").dt.date()).len().iter_rows()
    }


def main():
    args=_parse_args()
    today=date.today()
    start=args.start or (today.replace(year=today.year-args.years_back))
    end=args.end or (today - timedelta(days=1))
    calendar=NSECalendar.from_yaml(args.calendar)
    counts=_existing_counts()
    no_data=_known_no_data()

    missing=[
        d for d in calendar.trading_days(start,end)
        if d not in no_data and counts.get(d,0) < 350
    ]
    print(f"Required download days: {len(missing)}")
    if not missing:
        print("Source data is complete for the requested range.")
        return

    client=create_fyers_client(args.auth_file)
    cfg=FyersConfig()
    for d in missing:
        print(f"Fetching {d}...")
        frame=fetch_window(client,d,d,cfg)
        if frame.is_empty():
            no_data.add(d)
            continue
        out=RAW_DIR / f"banknifty_1m_{d.isoformat()}.parquet"
        if out.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing raw snapshot: {out}"
            )
        frame.write_parquet(out)
        actual=frame.height
        if actual < cfg.min_complete_candles:
            print(f"WARNING: {d} returned only {actual} candles")
        print(f"Saved {actual} candles -> {out}")
    _save_no_data(no_data)


if __name__=="__main__":
    main()
