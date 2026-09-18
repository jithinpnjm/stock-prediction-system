from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from src.common.contracts import BarrierType, LabelConfig, MARKET_TIMEZONE, SESSION_CLOSE

IST = ZoneInfo(MARKET_TIMEZONE)


def _session_close(ts: datetime) -> datetime:
    local = ts.astimezone(IST)
    return local.replace(
        hour=SESSION_CLOSE.hour,
        minute=SESSION_CLOSE.minute,
        second=0,
        microsecond=0,
    )


def _first_touch(
    *,
    entry: float,
    high: np.ndarray,
    low: np.ndarray,
    timestamps: np.ndarray,
    target: float,
    stop: float,
) -> tuple[str, object]:
    target_is_above = target > entry
    stop_is_above = stop > entry

    for j in range(len(high)):
        target_hit = high[j] >= target if target_is_above else low[j] <= target
        stop_hit = high[j] >= stop if stop_is_above else low[j] <= stop
        if target_hit and stop_hit:
            return BarrierType.AMBIGUOUS.value, timestamps[j]
        if target_hit:
            kind = (
                BarrierType.LONG_TARGET.value
                if target_is_above
                else BarrierType.SHORT_TARGET.value
            )
            return kind, timestamps[j]
        if stop_hit:
            kind = (
                BarrierType.LONG_STOP.value
                if not stop_is_above
                else BarrierType.SHORT_STOP.value
            )
            return kind, timestamps[j]
    return BarrierType.TIME.value, timestamps[-1] if len(timestamps) else None


def apply_triple_barrier_labels(
    events: pl.DataFrame,
    one_minute: pl.DataFrame,
    config: LabelConfig | None = None,
) -> pl.DataFrame:
    config = config or LabelConfig()
    one = one_minute.sort("timestamp")
    one_ts = np.asarray(one.get_column("timestamp").to_list(), dtype=object)
    one_high = one.get_column("high").to_numpy()
    one_low = one.get_column("low").to_numpy()

    results: list[dict[str, object]] = []
    event_ts = events.get_column("timestamp").to_list()
    event_close = events.get_column("close").to_numpy()

    for i, start in enumerate(event_ts):
        entry_time = start + timedelta(minutes=config.entry_delay_minutes)
        expiry = min(
            entry_time + timedelta(minutes=5 * config.horizon_bars),
            _session_close(entry_time),
        )
        left = int(np.searchsorted(one_ts, entry_time, side="right"))
        right = int(np.searchsorted(one_ts, expiry, side="right"))

        if right <= left:
            results.append(
                {
                    "event_start": start,
                    "event_end": expiry,
                    "entry_time": entry_time,
                    "label": 0,
                    "barrier_type": BarrierType.TIME.value,
                    "barrier_time": None,
                    "path_complete": False,
                    "long_outcome": 0,
                    "short_outcome": 0,
                }
            )
            continue

        h = one_high[left:right]
        l = one_low[left:right]
        t = one_ts[left:right]
        entry = float(event_close[i])
        long_type, long_time = _first_touch(
            entry=entry,
            high=h,
            low=l,
            timestamps=t,
            target=entry + config.target_points,
            stop=entry - config.stop_points,
        )
        short_type, short_time = _first_touch(
            entry=entry,
            high=h,
            low=l,
            timestamps=t,
            target=entry - config.target_points,
            stop=entry + config.stop_points,
        )

        ambiguous = long_type == BarrierType.AMBIGUOUS.value or short_type == BarrierType.AMBIGUOUS.value
        long_target = long_type == BarrierType.LONG_TARGET.value
        short_target = short_type == BarrierType.SHORT_TARGET.value

        if ambiguous:
            label = 0
            barrier_type = BarrierType.AMBIGUOUS.value
            barrier_time = long_time if long_type == BarrierType.AMBIGUOUS.value else short_time
        elif long_target and short_target and long_time == short_time:
            label = 0
            barrier_type = BarrierType.AMBIGUOUS.value
            barrier_time = long_time
        elif long_target and (not short_target or long_time < short_time):
            label = 1
            barrier_type = BarrierType.LONG_TARGET.value
            barrier_time = long_time
        elif short_target:
            label = -1
            barrier_type = BarrierType.SHORT_TARGET.value
            barrier_time = short_time
        else:
            label = 0
            if long_type == BarrierType.LONG_STOP.value:
                barrier_type = long_type
                barrier_time = long_time
            elif short_type == BarrierType.SHORT_STOP.value:
                barrier_type = short_type
                barrier_time = short_time
            else:
                barrier_type = BarrierType.TIME.value
                barrier_time = long_time or short_time

        long_outcome = 1 if long_target else -1 if long_type == BarrierType.LONG_STOP.value else 0
        short_outcome = 1 if short_target else -1 if short_type == BarrierType.SHORT_STOP.value else 0

        complete = one_ts[right - 1] >= expiry - timedelta(minutes=1)
        results.append(
            {
                "event_start": start,
                "event_end": expiry,
                "entry_time": entry_time,
                "label": label,
                "barrier_type": barrier_type,
                "barrier_time": barrier_time,
                "path_complete": bool(complete),
                "long_outcome": long_outcome,
                "short_outcome": short_outcome,
            }
        )

    return events.join(
        pl.DataFrame(results),
        left_on="timestamp",
        right_on="event_start",
        how="left",
    ).drop("event_start")
