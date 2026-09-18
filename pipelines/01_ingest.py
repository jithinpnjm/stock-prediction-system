from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from src.data.calendar import NSECalendar
from src.data.exclusions import apply_known_exclusions
from src.data.ingest import read_source
from src.data.schemas import validate_required_schema


def run():
    source = os.getenv("RAW_SOURCE_PATH", "data/raw/fyers")
    legacy_candidates = [
        Path("data/banknifty_spot_1m.csv"),
        Path("data/raw/1m_data.parquet"),
    ]
    try:
        df = read_source(source)
    except FileNotFoundError:
        for candidate in legacy_candidates:
            if candidate.exists():
                df = read_source(candidate)
                break
        else:
            raise
    validate_required_schema(df)

    # Drop today's still-in-progress session: it cannot be a complete
    # trading day until the market closes, so it must not enter the
    # canonical dataset yet.
    today_ist = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    df = df.filter(pl.col("source_timestamp").dt.date() < today_ist)

    df = apply_known_exclusions(df)

    # Vendor feeds sometimes include a handful of pre-/post-market
    # indicative ticks outside 09:15-15:30. These are not part of the
    # canonical session by definition, so drop them here rather than
    # let them fail whole-dataset validation downstream.
    calendar = NSECalendar.from_yaml("configs/data/nse_holidays.yaml")
    df = calendar.filter_session(df)

    out = Path(os.getenv("BRONZE_PATH", "data/bronze/validated_1m.parquet"))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    print(f"ingested {df.height} rows -> {out}")


if __name__ == "__main__":
    run()
