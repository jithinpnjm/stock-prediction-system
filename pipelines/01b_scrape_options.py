"""Backfill 5 years of NSE F&O bhavcopy data for Bank Nifty (futures OI,
options OI/PCR/max-pain) -- a genuinely new, non-price-action data source.
Resumable: skips dates already present in the output parquet."""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl

from src.data.nse_fo_bhavcopy import fetch_bhavcopy, parse_banknifty_daily

OUT_PATH = Path("data/bronze/banknifty_fo_daily.parquet")
SLEEP_SECONDS = 0.6


def run():
    trading_dates = (
        pl.read_parquet("data/silver/5m_canonical.parquet")
        .select(pl.col("session_date").unique().sort())
        .to_series()
        .to_list()
    )

    existing = pl.DataFrame(schema={"date": pl.Date})
    if OUT_PATH.exists():
        existing = pl.read_parquet(OUT_PATH)
    done_dates = set(existing["date"].to_list()) if existing.height else set()

    rows = existing.to_dicts()
    n_new, n_missing, n_err = 0, 0, 0
    for i, d in enumerate(trading_dates):
        if d in done_dates:
            continue
        try:
            result = fetch_bhavcopy(d)
            if result.raw is None:
                n_missing += 1
                continue
            parsed = parse_banknifty_daily(result.raw, d)
            if parsed is None:
                n_missing += 1
                continue
            rows.append(parsed)
            n_new += 1
        except Exception as exc:  # noqa: BLE001
            print(f"{d}: FAILED {exc}")
            n_err += 1
        if (i + 1) % 50 == 0:
            print(
                f"progress: {i + 1}/{len(trading_dates)} new={n_new} missing={n_missing} err={n_err}"
            )
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(rows).sort("date").write_parquet(OUT_PATH)
        time.sleep(SLEEP_SECONDS)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = pl.DataFrame(rows).sort("date")
    out.write_parquet(OUT_PATH)
    print(f"done: total_rows={out.height} new={n_new} missing={n_missing} err={n_err}")


if __name__ == "__main__":
    run()
