from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from src.common.contracts import MARKET_TIMEZONE, SESSION_CLOSE, LabelConfig


def _session_close(ts: datetime) -> datetime:
    local = ts.astimezone(ZoneInfo(MARKET_TIMEZONE))
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

    event_ts = events.get_column("timestamp").to_list()
    prices = one_minute.sort("timestamp")
    ts = np.array(prices.get_column("timestamp").to_list(), dtype=object)
    highs = prices.get_column("high").to_numpy()
    lows = prices.get_column("low").to_numpy()

    mfe = np.full(len(event_ts), np.nan)
    mae = np.full(len(event_ts), np.nan)
    mfe_time = [None] * len(event_ts)
    mae_time = [None] * len(event_ts)

    for i, start in enumerate(event_ts):
        expiry = min(
            start + timedelta(minutes=5 * horizon_bars),
            _session_close(start),
        )
        left = int(np.searchsorted(ts, start, side="right"))
        right = int(np.searchsorted(ts, expiry, side="right"))
        if right <= left:
            continue
        h = highs[left:right]
        l = lows[left:right]
        if direction == 1:
            favorable = h - float(events["close"][i])
            adverse = float(events["close"][i]) - l
        else:
            favorable = float(events["close"][i]) - l
            adverse = h - float(events["close"][i])
        mfe[i] = max(0.0, float(np.nanmax(favorable)))
        mae[i] = max(0.0, float(np.nanmax(adverse)))
        mfe_time[i] = ts[left + int(np.nanargmax(favorable))]
        mae_time[i] = ts[left + int(np.nanargmax(adverse))]

    return events.with_columns(
        [
            pl.Series(f"mfe_{'long' if direction == 1 else 'short'}", mfe),
            pl.Series(f"mae_{'long' if direction == 1 else 'short'}", mae),
            pl.Series(f"mfe_time_{'long' if direction == 1 else 'short'}", mfe_time),
            pl.Series(f"mae_time_{'long' if direction == 1 else 'short'}", mae_time),
        ]
    )
