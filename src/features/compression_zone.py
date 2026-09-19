"""Causal compression-state and swing-zone-distance features.

Built from the event-study finding in src/research/move_reverse_engineering.py:
across the real 5-year history, 83% of >=200pt moves were preceded by
compression (low directional efficiency) in the prior bars, and 46%
started within 0.5xATR of a recent swing level or prior-day
high/low/close. Both features here are strictly causal: the swing
distance only uses a pivot once it has been *confirmed* by a later bar
(the same lag a live zigzag indicator would have), and compression
uses a trailing window ending at the current bar. Distances are
expressed in ATR units, not raw points, because Bank Nifty's price
level drifted ~36,000->57,000 over the 5 years and a fixed-point
"near a zone" tolerance would silently mean something different in
2021 vs 2026 (see docs/trading-research-plan.md session notes).
"""

from __future__ import annotations

import numpy as np
import polars as pl


def _causal_swing_distance(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    threshold_points: float,
) -> np.ndarray:
    """For each bar i, the distance from close[i] to the most recently
    *confirmed* zigzag swing price, using only bars <= i. NaN before
    any swing has been confirmed in this session."""
    n = len(high)
    out = np.full(n, np.nan)
    if n == 0:
        return out

    direction: str | None = None
    anchor_price = float((high[0] + low[0]) / 2)
    extreme_price = anchor_price
    last_confirmed_price = np.nan

    for i in range(1, n):
        out[i] = last_confirmed_price
        if direction is None:
            if high[i] - anchor_price >= threshold_points:
                direction = "up"
                extreme_price = high[i]
            elif anchor_price - low[i] >= threshold_points:
                direction = "down"
                extreme_price = low[i]
        elif direction == "up":
            if high[i] > extreme_price:
                extreme_price = high[i]
            elif extreme_price - low[i] >= threshold_points:
                last_confirmed_price = extreme_price
                anchor_price = extreme_price
                direction = "down"
                extreme_price = low[i]
        elif direction == "down":
            if low[i] < extreme_price:
                extreme_price = low[i]
            elif high[i] - extreme_price >= threshold_points:
                last_confirmed_price = extreme_price
                anchor_price = extreme_price
                direction = "up"
                extreme_price = high[i]
        out[i] = last_confirmed_price
    return out


def add_compression_zone_features(
    df: pl.DataFrame,
    efficiency_windows: tuple[int, ...] = (5, 10, 20),
    swing_threshold_points: float = 200.0,
) -> pl.DataFrame:
    out = df.sort("timestamp").with_columns(pl.col("timestamp").dt.date().alias("_session_date"))

    for n in efficiency_windows:
        net_move = (pl.col("close") - pl.col("close").shift(n)).over("_session_date")
        gross_range = (pl.col("high") - pl.col("low")).rolling_sum(n).over("_session_date")
        out = out.with_columns(
            (net_move.abs() / (gross_range + 1e-9)).alias(f"f_directional_efficiency_{n}"),
        )
        out = out.with_columns(
            (pl.col(f"f_directional_efficiency_{n}") < 0.4)
            .cast(pl.Int8)
            .alias(f"f_is_compressed_{n}"),
        )

    if "f_atr_14" not in out.columns:
        from .volatility import add_volatility_features

        out = add_volatility_features(out, periods=(14,))
    if "_session_date" not in out.columns:
        out = out.with_columns(pl.col("timestamp").dt.date().alias("_session_date"))

    swing_dist_frames = []
    for d in out["_session_date"].unique(maintain_order=False).sort().to_list():
        day = out.filter(pl.col("_session_date") == d)
        dist = _causal_swing_distance(
            day["high"].to_numpy(),
            day["low"].to_numpy(),
            day["close"].to_numpy(),
            swing_threshold_points,
        )
        swing_dist_frames.append(
            day.select("timestamp").with_columns(pl.Series("_swing_dist_points", dist))
        )
    swing_dist = pl.concat(swing_dist_frames)
    out = out.join(swing_dist, on="timestamp", how="left")
    out = out.with_columns(
        (pl.col("_swing_dist_points") / (pl.col("f_atr_14") + 1e-9)).alias(
            "f_dist_to_recent_swing_atr"
        )
    )
    return out.drop(["_session_date", "_swing_dist_points"])
