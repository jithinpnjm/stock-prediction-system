from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from src.common.contracts import MARKET_TIMEZONE, SESSION_CLOSE

IST = ZoneInfo(MARKET_TIMEZONE)


def _session_close(ts: datetime) -> datetime:
    local = ts.astimezone(IST)
    return local.replace(
        hour=SESSION_CLOSE.hour,
        minute=SESSION_CLOSE.minute,
        second=0,
        microsecond=0,
    )


def compute_mfe_mae(
    events: pl.DataFrame,
    one_minute: pl.DataFrame,
    *,
    horizon_bars: int = 75,
    direction: int = 1,
) -> pl.DataFrame:
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")

    event_ts = (
        events.get_column("entry_time")
        if "entry_time" in events.columns
        else events.get_column("timestamp")
    ).to_list()
    entry_prices = (
        events.get_column("entry_price")
        if "entry_price" in events.columns
        else events.get_column("close")
    ).to_numpy()

    prices = one_minute.sort("timestamp")
    ts = np.asarray(prices.get_column("timestamp").to_list(), dtype=object)
    highs = prices.get_column("high").to_numpy()
    lows = prices.get_column("low").to_numpy()

    mfe = np.full(len(event_ts), np.nan)
    mae = np.full(len(event_ts), np.nan)
    mfe_time = [None] * len(event_ts)
    mae_time = [None] * len(event_ts)

    for i, start in enumerate(event_ts):
        if entry_prices[i] is None:
            continue
        expiry = min(
            start + timedelta(minutes=5 * horizon_bars),
            _session_close(start),
        )
        left = int(np.searchsorted(ts, start, side="left"))
        right = int(np.searchsorted(ts, expiry, side="right"))
        if right <= left:
            continue

        h = highs[left:right]
        l = lows[left:right]
        entry = float(entry_prices[i])

        if direction == 1:
            favorable = h - entry
            adverse = entry - l
        else:
            favorable = entry - l
            adverse = h - entry

        mfe[i] = max(0.0, float(np.nanmax(favorable)))
        mae[i] = max(0.0, float(np.nanmax(adverse)))
        mfe_time[i] = ts[left + int(np.nanargmax(favorable))]
        mae_time[i] = ts[left + int(np.nanargmax(adverse))]

    suffix = "long" if direction == 1 else "short"
    return events.with_columns(
        [
            pl.Series(f"mfe_{suffix}", mfe),
            pl.Series(f"mae_{suffix}", mae),
            pl.Series(f"mfe_time_{suffix}", mfe_time),
            pl.Series(f"mae_time_{suffix}", mae_time),
        ]
    )
