from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from numba import njit


@njit
def _label_path(
    entry_prices,
    high,
    low,
    start_idx,
    end_idx,
    target_pts,
    stop_pts,
):
    n = len(start_idx)
    labels = np.zeros(n, dtype=np.int8)
    outcome_codes = np.zeros(n, dtype=np.int8)
    hit_idx = np.full(n, -1, dtype=np.int64)
    mfe_l = np.zeros(n)
    mae_l = np.zeros(n)
    mfe_s = np.zeros(n)
    mae_s = np.zeros(n)

    for i in range(n):
        entry = entry_prices[i]
        long_target = entry + target_pts
        long_stop = entry - stop_pts
        short_target = entry - target_pts
        short_stop = entry + stop_pts

        long_target_idx = -1
        long_stop_idx = -1
        short_target_idx = -1
        short_stop_idx = -1
        ambiguous_idx = -1

        mfe_long = 0.0
        mae_long = 0.0
        mfe_short = 0.0
        mae_short = 0.0

        for j in range(start_idx[i], end_idx[i]):
            up = high[j] - entry
            down = entry - low[j]
            mfe_long = max(mfe_long, up)
            mae_long = max(mae_long, down)
            mfe_short = max(mfe_short, down)
            mae_short = max(mae_short, up)

            if long_target_idx == -1 and long_stop_idx == -1:
                if high[j] >= long_target and low[j] <= long_stop:
                    long_stop_idx = j
                    ambiguous_idx = j if ambiguous_idx == -1 else ambiguous_idx
                elif high[j] >= long_target:
                    long_target_idx = j
                elif low[j] <= long_stop:
                    long_stop_idx = j

            if short_target_idx == -1 and short_stop_idx == -1:
                if low[j] <= short_target and high[j] >= short_stop:
                    short_stop_idx = j
                    ambiguous_idx = j if ambiguous_idx == -1 else ambiguous_idx
                elif low[j] <= short_target:
                    short_target_idx = j
                elif high[j] >= short_stop:
                    short_stop_idx = j

        long_valid = (
            long_target_idx != -1
            and (long_stop_idx == -1 or long_target_idx < long_stop_idx)
        )
        short_valid = (
            short_target_idx != -1
            and (short_stop_idx == -1 or short_target_idx < short_stop_idx)
        )

        if long_valid and short_valid:
            if long_target_idx <= short_target_idx:
                labels[i] = 1
                outcome_codes[i] = 1
                hit_idx[i] = long_target_idx
            else:
                labels[i] = -1
                outcome_codes[i] = -1
                hit_idx[i] = short_target_idx
        elif long_valid:
            labels[i] = 1
            outcome_codes[i] = 1
            hit_idx[i] = long_target_idx
        elif short_valid:
            labels[i] = -1
            outcome_codes[i] = -1
            hit_idx[i] = short_target_idx
        elif ambiguous_idx != -1:
            outcome_codes[i] = 2
            hit_idx[i] = ambiguous_idx

        mfe_l[i] = mfe_long
        mae_l[i] = mae_long
        mfe_s[i] = mfe_short
        mae_s[i] = mae_short

    return (
        labels,
        outcome_codes,
        hit_idx,
        mfe_l,
        mae_l,
        mfe_s,
        mae_s,
    )


def apply_triple_barrier_labels(
    events_5m: pl.DataFrame,
    bars_1m: pl.DataFrame,
    *,
    target_pts: float = 200.0,
    stop_pts: float = 70.0,
    max_horizon_minutes: int = 375,
    direction: Literal["both", "long", "short"] = "both",
) -> pl.DataFrame:
    if events_5m.is_empty() or bars_1m.is_empty():
        raise ValueError("events_5m and bars_1m must be non-empty")
    if "close" not in events_5m.columns:
        raise ValueError("events_5m must contain close for event-close labels")
    if target_pts <= 0 or stop_pts <= 0 or max_horizon_minutes <= 0:
        raise ValueError("target, stop and horizon must be positive")

    events = events_5m.sort("timestamp")
    bars = bars_1m.sort("timestamp")
    e_ts = events["timestamp"].dt.epoch(time_unit="ns").to_numpy()
    e_close = events["close"].to_numpy().astype(float)
    b_ts = bars["timestamp"].dt.epoch(time_unit="ns").to_numpy()
    e_dates = events["timestamp"].dt.date().to_list()
    b_dates = bars["timestamp"].dt.date().to_list()

    starts = np.searchsorted(b_ts, e_ts, side="right").astype(np.int64)
    ends = np.empty(len(e_ts), dtype=np.int64)
    horizon_ns = np.int64(max_horizon_minutes) * 60 * 1_000_000_000

    for i, ts in enumerate(e_ts):
        start_idx = int(starts[i])
        end_idx = int(np.searchsorted(b_ts, ts + horizon_ns, side="right"))
        while end_idx > start_idx and b_dates[end_idx - 1] != e_dates[i]:
            end_idx -= 1
        ends[i] = end_idx

    usable = ends > starts
    if not np.any(usable):
        raise ValueError("No events have a usable future source path")

    events = events.filter(pl.Series(usable))
    e_close = e_close[usable]
    starts = starts[usable]
    ends = ends[usable]

    (
        labels,
        outcome_codes,
        hit_idx,
        mfe_l,
        mae_l,
        mfe_s,
        mae_s,
    ) = _label_path(
        e_close,
        bars["high"].to_numpy(),
        bars["low"].to_numpy(),
        starts,
        ends,
        target_pts,
        stop_pts,
    )

    expiry_idx = ends - 1
    ns_dtype = pl.Datetime("ns", time_zone="Asia/Kolkata")

    event_end = pl.Series(
        "event_end_timestamp",
        b_ts[expiry_idx],
    ).cast(ns_dtype)

    barrier_values = np.full(len(labels), 0, dtype=np.int64)
    hit_mask = outcome_codes != 0
    barrier_values[hit_mask] = b_ts[hit_idx[hit_mask]]
    barrier = pl.Series("barrier_timestamp", barrier_values).cast(ns_dtype)
    barrier = barrier.set_at_idx(
        pl.Series(np.flatnonzero(~hit_mask)),
        None,
    )

    barrier_type = np.select(
        [
            outcome_codes == 1,
            outcome_codes == -1,
            outcome_codes == 2,
        ],
        [
            "target_long",
            "target_short",
            "ambiguous",
        ],
        default="no_target",
    )

    result = events.with_columns(
        pl.Series("entry_price", e_close),
        pl.Series("label", labels, dtype=pl.Int8),
        event_end,
        barrier,
        pl.Series("barrier_type", barrier_type),
        pl.Series("mfe_long_points", mfe_l),
        pl.Series("mae_long_points", mae_l),
        pl.Series("mfe_short_points", mfe_s),
        pl.Series("mae_short_points", mae_s),
        pl.lit(target_pts).alias("target_points"),
        pl.lit(stop_pts).alias("stop_points"),
        pl.lit(max_horizon_minutes).alias("max_horizon_minutes"),
    )

    if direction == "long":
        result = result.with_columns(
            pl.when(pl.col("label") == 1)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("label")
        )
    elif direction == "short":
        result = result.with_columns(
            pl.when(pl.col("label") == -1)
            .then(-1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("label")
        )

    from .mfe_mae import add_excursion_features
    from .timing import add_label_timing

    result = add_excursion_features(result)
    result = add_label_timing(result, "barrier_timestamp")
    return result
