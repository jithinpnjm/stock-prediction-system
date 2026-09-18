from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.data.calendar import NSECalendar
from src.data.exclusions import load_excluded_dates
from src.data.validate import assert_valid


def run():
    df = pl.read_parquet("data/bronze/validated_1m.parquet")
    calendar = NSECalendar.from_yaml("configs/data/nse_holidays.yaml")
    # Days dropped by pipelines/01_ingest.py (abbreviated special
    # sessions, vendor gaps) are deliberately absent from bronze, not
    # missing data -- treat them as non-trading days here too so
    # completeness checks don't flag them.
    calendar = NSECalendar(
        holidays=calendar.holidays | load_excluded_dates(),
        timezone=calendar.timezone,
    )
    reports = assert_valid(df, calendar=calendar)
    out = Path("data/bronze/validation_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2) + "\n")
    print("validation passed")


if __name__ == "__main__":
    run()
