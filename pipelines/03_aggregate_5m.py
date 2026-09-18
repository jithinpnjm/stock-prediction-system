from __future__ import annotations

from pathlib import Path

import polars as pl

from src.data.aggregate import aggregate_1m_to_5m, validate_aggregation
from src.data.calendar import NSECalendar


def run():
    df = pl.read_parquet("data/bronze/validated_1m.parquet")
    calendar = NSECalendar.from_yaml("configs/data/nse_holidays.yaml")
    df5 = aggregate_1m_to_5m(df, calendar=calendar, drop_incomplete=True)
    validate_aggregation(df5)
    out = Path("data/silver/5m_canonical.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df5.write_parquet(out)
    print(f"created {df5.height} canonical 5m rows")


if __name__ == "__main__":
    run()
