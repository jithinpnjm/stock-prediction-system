from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from numba import njit

from src.common.contracts import (
    BarrierType,
    LabelConfig,
    MARKET_TIMEZONE,
)

IST = ZoneInfo(MARKET_TIMEZONE)

_BARRIER_TIME = 0
_LONG_TARGET = 1
_LONG_STOP = 2
_SHORT_TARGET = 3
_SHORT_STOP = 4
_AMBIGUOUS = 5


@njit(cache=True)
def _scan_paths(
    one_epoch: np.ndarray,
    one_session: np.ndarray,
    one_open: np.ndarray,
    one_high: np.ndarray,
    one_low: np.ndarray,
    event_session: np.ndarray,
    entry_indices: np.ndarray,
    expiry_epoch: np.ndarray,
    target_points: float,
    stop_points: float,
):
    n = len(entry_indices)
    labels = np.zeros(n, dtype=np.int8)
    barrier_codes = np.zeros(n, dtype=np.int8)
    barrier_indices = np.full(n, -1, dtype=np.int64)
    long_outcomes = np.zeros(n, dtype=np.int8)
    short_outcomes = np.zeros(n, dtype=np.int8)
    entry_prices = np.full(n, np.nan, dtype=np.float64)
    path_complete = np.zeros(n, dtype=np.int8)

    for i in range(n):
        start = entry_indices[i]

        if start < 0 or start >= len(one_epoch):
            continue

        if one_session[start] != event_session[i]:
            continue

        entry = one_open[start]
        entry_prices[i] = entry

        end = start
        while (
            end < len(one_epoch)
            and one_epoch[end] <= expiry_epoch[i]
            and one_session[end] == event_session[i]
        ):
            end += 1

        if end <= start:
            continue

        long_state = 0
        short_state = 0
        long_idx = -1
        short_idx = -1

        long_target = entry + target_points
        long_stop = entry - stop_points
        short_target = entry - target_points
        short_stop = entry + stop_points

        for j in range(start, end):
            if long_state == 0:
                hit_target = one_high[j] >= long_target
                hit_stop = one_low[j] <= long_stop
                if hit_target and hit_stop:
                    long_state = 3
                    long_idx = j
                elif hit_target:
                    long_state = 1
                    long_idx = j
                elif hit_stop:
                    long_state = 2
                    long_idx = j

            if short_state == 0:
                hit_target = one_low[j] <= short_target
                hit_stop = one_high[j] >= short_stop
                if hit_target and hit_stop:
                    short_state = 3
                    short_idx = j
                elif hit_target:
                    short_state = 1
                    short_idx = j
                elif hit_stop:
                    short_state = 2
                    short_idx = j

            if long_state != 0 and short_state != 0:
                break

        if long_state == 1:
            long_outcomes[i] = 1
        elif long_state == 2:
            long_outcomes[i] = -1

        if short_state == 1:
            short_outcomes[i] = 1
        elif short_state == 2:
            short_outcomes[i] = -1

        if long_state == 3 or short_state == 3:
            labels[i] = 0
            barrier_codes[i] = _AMBIGUOUS
            barrier_indices[i] = (
                long_idx if long_state == 3 else short_idx
            )
        elif long_state == 1 and short_state == 1:
            if long_idx < short_idx:
                labels[i] = 1
                barrier_codes[i] = _LONG_TARGET
                barrier_indices[i] = long_idx
            elif short_idx < long_idx:
                labels[i] = -1
                barrier_codes[i] = _SHORT_TARGET
                barrier_indices[i] = short_idx
            else:
                labels[i] = 0
                barrier_codes[i] = _AMBIGUOUS
                barrier_indices[i] = long_idx
        elif long_state == 1:
            labels[i] = 1
            barrier_codes[i] = _LONG_TARGET
            barrier_indices[i] = long_idx
        elif short_state == 1:
            labels[i] = -1
            barrier_codes[i] = _SHORT_TARGET
            barrier_indices[i] = short_idx
        elif long_state == 2:
            barrier_codes[i] = _LONG_STOP
            barrier_indices[i] = long_idx
        elif short_state == 2:
            barrier_codes[i] = _SHORT_STOP
            barrier_indices[i] = short_idx
        else:
            barrier_codes[i] = _BARRIER_TIME
            barrier_indices[i] = end - 1

        path_complete[i] = int(
            one_epoch[end - 1] >= expiry_epoch[i] - 60
        )

    return (
        labels,
        barrier_codes,
        barrier_indices,
        long_outcomes,
        short_outcomes,
        entry_prices,
        path_complete,
    )


def _session_close(ts):
    local = ts.astimezone(IST)
    return local.replace(
        hour=15,
        minute=30,
        second=0,
        microsecond=0,
    )


def apply_triple_barrier_labels(
    events: pl.DataFrame,
    one_minute: pl.DataFrame,
    config: LabelConfig | None = None,
) -> pl.DataFrame:
    config = config or LabelConfig()

    if events.is_empty():
        raise ValueError("events is empty")
    if one_minute.is_empty():
        raise ValueError("one_minute is empty")

    events = events.sort("timestamp")
    one = one_minute.sort("timestamp")

    event_times = events.get_column(
        "timestamp"
    ).to_list()

    one_timestamp = one.get_column("timestamp")
    one_epoch = one_timestamp.dt.epoch("s").to_numpy()
    one_session = np.asarray(
        [
            value.toordinal()
            for value in one.get_column(
                "session_date"
            ).to_list()
        ],
        dtype=np.int64,
    )

    event_session = np.asarray(
        [
            value.toordinal()
            for value in events.get_column(
                "session_date"
            ).to_list()
        ],
        dtype=np.int64,
    )

    entry_times = [
        value + timedelta(
            minutes=config.entry_delay_minutes
        )
        for value in event_times
    ]

    session_last: dict[int, int] = {}
    for idx, session in enumerate(one_session):
        session_last[session] = idx

    expiry_times = []
    for entry, session_key in zip(
        entry_times,
        event_session,
    ):
        last_idx = session_last.get(
            int(session_key),
            -1,
        )
        if last_idx < 0:
            expiry_times.append(entry)
            continue

        session_close = one_timestamp[last_idx]
        expiry_times.append(
            min(
                entry + timedelta(
                    minutes=5 * config.horizon_bars
                ),
                session_close,
            )
        )

    entry_epoch = np.asarray(
        [
            int(value.timestamp())
            for value in entry_times
        ],
        dtype=np.int64,
    )
    expiry_epoch = np.asarray(
        [
            int(value.timestamp())
            for value in expiry_times
        ],
        dtype=np.int64,
    )

    entry_indices = np.searchsorted(
        one_epoch,
        entry_epoch,
        side="left",
    )

    (
        labels,
        barrier_codes,
        barrier_indices,
        long_outcomes,
        short_outcomes,
        entry_prices,
        path_complete,
    ) = _scan_paths(
        one_epoch,
        one_session,
        one.get_column("open").to_numpy(),
        one.get_column("high").to_numpy(),
        one.get_column("low").to_numpy(),
        event_session,
        entry_indices,
        expiry_epoch,
        float(config.target_points),
        float(config.stop_points),
    )

    barrier_epoch = np.full(
        len(barrier_indices),
        -1,
        dtype=np.int64,
    )
    valid = barrier_indices >= 0
    barrier_epoch[valid] = one_epoch[
        barrier_indices[valid]
    ]

    barrier_type_map = {
        _BARRIER_TIME: BarrierType.TIME.value,
        _LONG_TARGET: BarrierType.LONG_TARGET.value,
        _LONG_STOP: BarrierType.LONG_STOP.value,
        _SHORT_TARGET: BarrierType.SHORT_TARGET.value,
        _SHORT_STOP: BarrierType.SHORT_STOP.value,
        _AMBIGUOUS: BarrierType.AMBIGUOUS.value,
    }

    barrier_types = [
        barrier_type_map[int(code)]
        for code in barrier_codes
    ]

    barrier_time = (
        pl.Series(
            "barrier_time",
            barrier_epoch,
            dtype=pl.Int64,
        )
        .cast(
            pl.Datetime(
                time_unit="s",
                time_zone="UTC",
            )
        )
        .dt.convert_time_zone(
            MARKET_TIMEZONE
        )
    )
    barrier_time = pl.when(
        pl.Series(
            barrier_epoch >= 0
        )
    ).then(barrier_time).otherwise(None)

    label_frame = pl.DataFrame(
        {
            "event_start": event_times,
            "event_end": expiry_times,
            "entry_time": entry_times,
            "entry_price": entry_prices,
            "label": labels,
            "barrier_type": barrier_types,
            "path_complete": path_complete.astype(bool),
            "long_outcome": long_outcomes,
            "short_outcome": short_outcomes,
        }
    ).with_columns(
        barrier_time.alias(
            "barrier_time"
        )
    )

    return (
        events.join(
            label_frame,
            left_on="timestamp",
            right_on="event_start",
            how="left",
        )
        .drop("event_start")
    )
