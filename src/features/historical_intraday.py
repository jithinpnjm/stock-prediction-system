from __future__ import annotations

from collections import defaultdict, deque
import numpy as np
import polars as pl


def add_historical_intraday_context(
    df: pl.DataFrame,
    lookback_days: int = 20,
) -> pl.DataFrame:
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    out = df.sort("timestamp")
    if "f_session_bar_index" not in out.columns:
        from .time_features import add_time_features

        out = add_time_features(out)

    dates = out["timestamp"].dt.date().to_list()
    slots = out["f_session_bar_index"].to_numpy()
    close = out["close"].to_numpy()

    values = np.full(len(out), np.nan, dtype=float)
    stds = np.full(len(out), np.nan, dtype=float)
    rates = np.full(len(out), np.nan, dtype=float)
    history: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=lookback_days))
    current_slots: dict[int, float] = {}
    previous_date = None

    for i, (d, slot_value) in enumerate(zip(dates, slots, strict=False)):
        if previous_date is not None and d != previous_date:
            for slot, value in current_slots.items():
                history[slot].append(value)
            current_slots = {}

        slot = int(slot_value)
        if i > 0 and d == dates[i - 1]:
            current_return = float(close[i] / close[i - 1] - 1.0)
        else:
            current_return = float("nan")

        prior = list(history[slot])
        if prior:
            values[i] = float(np.mean(prior))
            stds[i] = float(np.std(prior, ddof=1)) if len(prior) > 1 else 0.0
            rates[i] = float(np.mean(np.asarray(prior) > 0))

        current_slots[slot] = current_return
        previous_date = d

    return out.with_columns(
        pl.Series("f_historical_slot_return_mean", values),
        pl.Series("f_historical_slot_return_std", stds),
        pl.Series("f_historical_slot_up_rate", rates),
    ).with_columns(
        pl.col("f_historical_slot_return_mean").fill_nan(None),
        pl.col("f_historical_slot_return_std").fill_nan(None),
        pl.col("f_historical_slot_up_rate").fill_nan(None),
    )
