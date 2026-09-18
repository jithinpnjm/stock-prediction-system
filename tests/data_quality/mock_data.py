from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from src.common.contracts import MARKET_TIMEZONE


def generate_mock_data(path: str = "data/raw/1m_data.parquet", sessions: int = 3) -> None:
    os = __import__("os")
    os.makedirs("data/raw", exist_ok=True)
    tz = ZoneInfo(MARKET_TIMEZONE)
    timestamps = []
    for offset in range(sessions):
        day = date(2026, 1, 5) + timedelta(days=offset)
        while day.weekday() >= 5:
            day += timedelta(days=1)
        start = datetime(day.year, day.month, day.day, 9, 15, tzinfo=tz)
        timestamps.extend(start + timedelta(minutes=i) for i in range(375))
    rng = np.random.default_rng(42)
    close = 40000 + np.cumsum(rng.normal(0, 5, len(timestamps)))
    open_prices = close - rng.normal(0, 2, len(timestamps))
    high = np.maximum(open_prices, close) + rng.uniform(0, 10, len(timestamps))
    low = np.minimum(open_prices, close) - rng.uniform(0, 10, len(timestamps))
    df = pl.DataFrame({"datetime": timestamps, "open": open_prices, "high": high, "low": low, "close": close, "volume": rng.integers(100, 5000, len(timestamps))})
    df.write_parquet(path)
    print(f"Generated {df.height} valid session-aware 1m rows at {path}")


if __name__ == "__main__":
    generate_mock_data()
