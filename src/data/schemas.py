from datetime import time

import polars as pl

RAW_1M_SCHEMA = {
    "datetime": pl.Datetime,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
}


def get_market_hours() -> tuple[time, time]:
    return time(9, 15), time(15, 30)
