from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from src.common.contracts import BarrierType, MARKET_TIMEZONE, SESSION_CLOSE, LabelConfig


def _session_close(ts: datetime) -> datetime:
    local = ts.astimezone(ZoneInfo(MARKET_TIMEZONE))
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
    target_hit: int | None = None
    stop_hit: int | None = None
    for j in range(len(high)):
        target_j = high[j] >= target if target > entry else low[j] <= target
        stop_j = low[j] <= stop if stop < entry else high[j] >= stop
        if target_j:
            target_hit = j
        if stop_j:
            stop_hit = j
        if target_hit is not None or stop_hit is not None:
            if target_hit is not None and stop_hit is not None and target_hit == stop_hit:
                return BarrierType.AMBIGUOUS.value, timestamps[j]
            if target_hit is not None and (stop_hit is None or target_hit < stop_hit):
                return BarrierType.LONG_TARGET.value if target > entry else BarrierType.SHORT_TARGET.value, timestamps[target_hit]
            return BarrierType.LONG_STOP.value if stop < entry else BarrierType.SHORT_STOP.value, timestamps[stop_hit]
    return BarrierType.TIME.value, timestamps[-1] if len(timestamps) else None


def apply_triple_barrier_labels(
    events: pl.DataFrame,
    one_minute: pl.DataFrame,
    config: LabelConfig | None = None,
) -> pl.DataFrame:
    config = config or LabelConfig()
    if events.is_empty() or one_minute.is_empty():
        raise ValueError("Both events and one-minute data are required")

    one = one_minute.sort("timestamp")
    one_ts = np.array(one.get_column("timestamp").to_list(), dtype=object)
    one_high = one.get_column("high").to_numpy()
    one_low = one.get_column("low").to_numpy()

    result: list[dict[str, object]] = []
    event_ts = events.get_column("timestamp").to_list()
    close = events.get_column("close").to_numpy()

    for i, start in enumerate(event_ts):
        entry_time = start + timedelta(minutes=config.entry_delay_minutes)
        expiry = min(
            entry_time + timedelta(minutes=5 * config.horizon_bars),
            _session_close(entry_time),
        )
        left = int(np.searchsorted(one_ts, entry_time, side="right"))
        right = int(np.searchsorted(one_ts, expiry, side="right"))

        if right <= left:
            result.append(
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
        entry = float(close[i])

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

        # Directional event label: only a clean target-first result becomes a
        # directional class. Any stop, timeout, or intrabar ambiguity is 0.
        first_candidates = []
        if long_type == BarrierType.LONG_TARGET.value:
            first_candidates.append((long_time, 1, long_type))
        if short_type == BarrierType.SHORT_TARGET.value:
            first_candidates.append((short_time, -1, short_type))

        if not first_candidates:
            label = 0
            barrier_type = (
                BarrierType.AMBIGUOUS.value
                if long_type == BarrierType.AMBIGUOUS.value or short_type == BarrierType.AMBIGUOUS.value
                else BarrierType.TIME.value
                if long_type == BarrierType.TIME.value and short_type == BarrierType.TIME.value
                else long_type if long_type in (BarrierType.LONG_STOP.value,) else short_type
            )
            barrier_time = long_time if long_type != BarrierType.TIME.value else short_time
        else:
            first_time, label, barrier_type = min(
                first_candidates, key=lambda x: x[0]
            )
            barrier_time = first_time

        long_outcome = 1 if long_type == BarrierType.LONG_TARGET.value else -1 if long_type == BarrierType.LONG_STOP.value else 0
        short_outcome = 1 if short_type == BarrierType.SHORT_TARGET.value else -1 if short_type == BarrierType.SHORT_STOP.value else 0

        result.append(
            {
                "event_start": start,
                "event_end": expiry,
                "entry_time": entry_time,
                "label": label,
                "barrier_type": barrier_type,
                "barrier_time": barrier_time,
                "path_complete": right < len(one_ts) and one_ts[right - 1] >= expiry - timedelta(minutes=1),
                "long_outcome": long_outcome,
                "short_outcome": short_outcome,
            }
        )

    labels = pl.DataFrame(result)
    return events.join(labels, left_on="timestamp", right_on="event_start", how="left").drop("event_start")


def add_path_statistics(
    labeled: pl.DataFrame,
    one_minute: pl.DataFrame,
    *,
    horizon_bars: int = 75,
) -> pl.DataFrame:
    out = compute_mfe_mae(labeled, one_minute, horizon_bars=horizon_bars, direction=1)
    return compute_mfe_mae(out, one_minute, horizon_bars=horizon_bars, direction=-1)
