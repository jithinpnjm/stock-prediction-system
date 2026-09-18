from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from numba import njit


@njit
def _label_path(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    start_idx: np.ndarray,
    end_idx: np.ndarray,
    event_ts_ns: np.ndarray,
    target_pts: float,
    stop_pts: float,
):
    n = len(event_ts_ns)
    labels = np.zeros(n, dtype=np.int8)
    barrier_ts = np.full(n, -1, dtype=np.int64)
    mfe_long = np.zeros(n, dtype=np.float64)
    mae_long = np.zeros(n, dtype=np.float64)
    mfe_short = np.zeros(n, dtype=np.float64)
    mae_short = np.zeros(n, dtype=np.float64)

    for i in range(n):
        entry = close[start_idx[i] - 1] if start_idx[i] > 0 else close[0]
        long_target = entry + target_pts
        long_stop = entry - stop_pts
        short_target = entry - target_pts
        short_stop = entry + stop_pts

        long_target_i = -1
        long_stop_i = -1
        short_target_i = -1
        short_stop_i = -1
        ml = 0.0
        al = 0.0
        ms = 0.0
        ass = 0.0

        for j in range(start_idx[i], end_idx[i]):
            up = high[j] - entry
            down = entry - low[j]
            if up > ml:
                ml = up
            if down > al:
                al = down
            if down > ms:
                ms = down
            if up > ass:
                ass = up

            # A single OHLC bar cannot reveal intrabar ordering. Resolve a
            # same-bar target/stop collision conservatively as a loss.
            long_both = high[j] >= long_target and low[j] <= long_stop
            short_both = low[j] <= short_target and high[j] >= short_stop

            if long_target_i == -1 and long_stop_i == -1:
                if long_both:
                    long_stop_i = j
                elif high[j] >= long_target:
                    long_target_i = j
                elif low[j] <= long_stop:
                    long_stop_i = j

            if short_target_i == -1 and short_stop_i == -1:
                if short_both:
                    short_stop_i = j
                elif low[j] <= short_target:
                    short_target_i = j
                elif high[j] >= short_stop:
                    short_stop_i = j

            if (
                long_target_i != -1
                and long_stop_i != -1
                and short_target_i != -1
                and short_stop_i != -1
            ):
                break

        long_valid = long_target_i != -1 and (
            long_stop_i == -1 or long_target_i < long_stop_i
        )
        short_valid = short_target_i != -1 and (
            short_stop_i == -1 or short_target_i < short_stop_i
        )

        if long_valid and short_valid:
            if long_target_i <= short_target_i:
                labels[i] = 1
                barrier_ts[i] = event_ts_ns[i] + 1
            else:
                labels[i] = -1
                barrier_ts[i] = event_ts_ns[i] + 1
        elif long_valid:
            labels[i] = 1
            barrier_ts[i] = event_ts_ns[i] + 1
        elif short_valid:
            labels[i] = -1
            barrier_ts[i] = event_ts_ns[i] + 1

        mfe_long[i] = ml
        mae_long[i] = al
        mfe_short[i] = ms
        mae_short[i] = ass

    return labels, barrier_ts, mfe_long, mae_long, mfe_short, mae_short


def apply_triple_barrier_labels(
    events_5m: pl.DataFrame,
    bars_1m: pl.DataFrame,
    *,
    target_pts: float = 200.0,
    stop_pts: float = 70.0,
    max_horizon_minutes: int = 375,
    direction: Literal["both", "long", "short"] = "both",
) -> pl.DataFrame:
    """Label completed 5m events using the subsequent 1m price path.

    The event timestamp is the information-available time. Only 1m bars with
    timestamps strictly after the event are examined. The horizon is capped at
    the end of the same trading session by the caller's 1m data.
    """
    if events_5m.is_empty() or bars_1m.is_empty():
        raise ValueError("Both event and source datasets must be non-empty")
    events = events_5m.sort("timestamp")
    bars = bars_1m.sort("timestamp")

    e_ts = events["timestamp"].dt.epoch(time_unit="ns").to_numpy()
    b_ts = bars["timestamp"].dt.epoch(time_unit="ns").to_numpy()
    e_dates = events["timestamp"].dt.date().to_list()
    b_dates = bars["timestamp"].dt.date().to_list()
    close = bars["close"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()

    starts = np.searchsorted(b_ts, e_ts, side="right").astype(np.int64)
    ends = np.empty(len(e_ts), dtype=np.int64)
    horizon_ns = np.int64(max_horizon_minutes) * 60 * 1_000_000_000

    # Restrict each event to the same session date and configured horizon.
    for i, ts in enumerate(e_ts):
        j = starts[i]
        candidate_end = np.searchsorted(b_ts, ts + horizon_ns, side="right")
        while candidate_end > j and b_dates[candidate_end - 1] != e_dates[i]:
            candidate_end -= 1
        ends[i] = max(j, candidate_end)

    valid = starts > 0
    if not np.all(valid):
        raise ValueError("Some event timestamps do not have a prior source close")

    labels, _, mfe_l, mae_l, mfe_s, mae_s = _label_path(
        close,
        high,
        low,
        starts,
        ends,
        e_ts,
        target_pts,
        stop_pts,
    )

    if direction == "long":
        labels = np.where(labels == 1, 1, 0).astype(np.int8)
    elif direction == "short":
        labels = np.where(labels == -1, -1, 0).astype(np.int8)

    return events.with_columns(
        pl.Series("label", labels, dtype=pl.Int8),
        pl.Series("mfe_long_points", mfe_l),
        pl.Series("mae_long_points", mae_l),
        pl.Series("mfe_short_points", mfe_s),
        pl.Series("mae_short_points", mae_s),
        pl.lit(target_pts).alias("target_points"),
        pl.lit(stop_pts).alias("stop_points"),
        pl.lit(max_horizon_minutes).alias("max_horizon_minutes"),
    )
