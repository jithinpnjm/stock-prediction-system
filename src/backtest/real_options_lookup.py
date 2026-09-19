"""Look up REAL Bank Nifty option premiums (5m OHLC, real traded prices)
for backtesting -- ground truth where available, replacing the Black-
Scholes simulation in options_pricing.py for the period this data
covers. Real premiums already embed actual IV, skew, theta, and
liquidity effects that a BS simulation can only approximate.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from functools import lru_cache

import polars as pl

DEFAULT_PATH = "data/bronze/options_5m_consolidated.parquet"
_EXPIRY_MONTH_CODE = {
    1: "JAN",
    2: "FEB",
    3: "MAR",
    4: "APR",
    5: "MAY",
    6: "JUN",
    7: "JUL",
    8: "AUG",
    9: "SEP",
    10: "OCT",
    11: "NOV",
    12: "DEC",
}


@lru_cache(maxsize=1)
def _load(path: str = DEFAULT_PATH) -> pl.DataFrame:
    return pl.read_parquet(path)


def data_coverage(path: str = DEFAULT_PATH) -> tuple[date, date] | None:
    p = Path(path)
    if not p.exists():
        return None
    df = _load(path)
    if df.height == 0:
        return None
    return df["date"].min(), df["date"].max()


def build_symbol(expiry_month: date, strike: float, option_type: str) -> str:
    """expiry_month: any date within the contract's expiry month (we only
    have monthly contracts). Fyers symbol format: NSE:BANKNIFTY{YY}{MON}{STRIKE}{CE/PE}"""
    yy = expiry_month.year % 100
    mon = _EXPIRY_MONTH_CODE[expiry_month.month]
    return f"NSE:BANKNIFTY{yy:02d}{mon}{int(strike)}{option_type}"


def lookup_premium(
    timestamp: datetime,
    symbol: str,
    path: str = DEFAULT_PATH,
    *,
    tolerance_bars: int = 3,
) -> float | None:
    """Nearest 5m close at/after `timestamp` for `symbol`, within
    `tolerance_bars` * 5 minutes. None if no match (illiquid strike,
    symbol not in this dataset, or outside its date coverage)."""
    df = _load(path)
    sub = df.filter(pl.col("fyers_symbol") == symbol).sort("datetime")
    if sub.height == 0:
        return None
    # nearest bar at-or-after timestamp, within tolerance
    after = sub.filter(pl.col("datetime") >= timestamp)
    if after.height == 0:
        return None
    row = after.head(1)
    delta = (row["datetime"][0] - timestamp).total_seconds()
    if delta > tolerance_bars * 300:
        return None
    return float(row["close"][0])
