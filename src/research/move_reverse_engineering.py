"""Event-study / reverse-engineering analysis of significant intraday moves.

This is exploratory chart analysis, NOT a causal feature pipeline. It
identifies every intraday swing-to-swing leg of >= `threshold_points`
on the 5-minute canonical series, then characterizes what preceded
each leg: proximity to prior-day high/low/close and recent swing
levels (in ATR units, since Bank Nifty's price level drifted
~36,000->57,000 over 5 years and a fixed-point "near a zone" tolerance
would silently mean something different in 2021 vs 2026), the
candle-cluster state in the bars immediately before the move started,
and whether the day as a whole was a trend day or a range/box day.

Any pattern found here is a hypothesis to test with a properly causal,
point-in-time feature -- not something to feed into a model directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class Pivot:
    idx: int
    timestamp: object
    price: float
    kind: str  # "high" or "low"


def find_session_pivots(
    timestamps: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    threshold_points: float,
) -> list[Pivot]:
    """Classic ZigZag pivot detection on one session's bars (in order)."""
    n = len(high)
    if n == 0:
        return []
    pivots: list[Pivot] = []
    direction: str | None = None
    anchor_price = float((high[0] + low[0]) / 2)
    extreme_price = anchor_price
    extreme_idx = 0

    for i in range(1, n):
        if direction is None:
            if high[i] - anchor_price >= threshold_points:
                direction = "up"
                extreme_price = high[i]
                extreme_idx = i
            elif anchor_price - low[i] >= threshold_points:
                direction = "down"
                extreme_price = low[i]
                extreme_idx = i
        elif direction == "up":
            if high[i] > extreme_price:
                extreme_price = high[i]
                extreme_idx = i
            elif extreme_price - low[i] >= threshold_points:
                pivots.append(
                    Pivot(extreme_idx, timestamps[extreme_idx], float(extreme_price), "high")
                )
                anchor_price = extreme_price
                direction = "down"
                extreme_price = low[i]
                extreme_idx = i
        elif direction == "down":
            if low[i] < extreme_price:
                extreme_price = low[i]
                extreme_idx = i
            elif high[i] - extreme_price >= threshold_points:
                pivots.append(
                    Pivot(extreme_idx, timestamps[extreme_idx], float(extreme_price), "low")
                )
                anchor_price = extreme_price
                direction = "up"
                extreme_price = high[i]
                extreme_idx = i

    if direction is not None:
        pivots.append(
            Pivot(
                extreme_idx,
                timestamps[extreme_idx],
                float(extreme_price),
                "high" if direction == "up" else "low",
            )
        )
    return pivots


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = np.full(len(tr), np.nan)
    for i in range(period - 1, len(tr)):
        atr[i] = tr[i - period + 1 : i + 1].mean()
    return atr


def analyze_moves(
    df5: pl.DataFrame,
    threshold_points: float = 200.0,
    cluster_lookback: int = 5,
    zone_atr_multiple: float = 0.5,
    recent_swing_lookback_days: int = 5,
) -> pl.DataFrame:
    """Identify every >= threshold_points intraday swing leg and
    characterize the conditions immediately preceding it."""
    df5 = df5.sort("timestamp").with_columns(pl.col("timestamp").dt.date().alias("_date"))
    dates = df5["_date"].unique(maintain_order=False).sort().to_list()

    # prior-day OHLC lookup
    daily = (
        df5.group_by("_date")
        .agg(
            pl.col("open").first().alias("day_open"),
            pl.col("high").max().alias("day_high"),
            pl.col("low").min().alias("day_low"),
            pl.col("close").last().alias("day_close"),
        )
        .sort("_date")
    )
    prev_day = {}
    prev_row = None
    for row in daily.iter_rows(named=True):
        if prev_row is not None:
            prev_day[row["_date"]] = prev_row
        prev_row = row

    close_all = df5["close"].to_numpy()
    high_all = df5["high"].to_numpy()
    low_all = df5["low"].to_numpy()
    atr_all = _atr(high_all, low_all, close_all, period=14)
    df5 = df5.with_columns(pl.Series("_atr14", atr_all))

    all_day_pivots: dict = {}  # date -> list[Pivot] (levels available for "recent swing" lookback)
    rows = []

    for d in dates:
        day = df5.filter(pl.col("_date") == d)
        ts = day["timestamp"].to_numpy()
        o = day["open"].to_numpy()
        h = day["high"].to_numpy()
        lo = day["low"].to_numpy()
        c = day["close"].to_numpy()
        atr = day["_atr14"].to_numpy()

        pivots = find_session_pivots(ts, h, lo, threshold_points)
        all_day_pivots[d] = pivots
        if len(pivots) < 2:
            continue

        # recent swing levels from the last N trading days (levels only, not this day's own pivots)
        recent_days = [dd for dd in dates if dd < d][-recent_swing_lookback_days:]
        recent_levels = [p.price for dd in recent_days for p in all_day_pivots.get(dd, [])]

        prev = prev_day.get(d)

        for k in range(len(pivots) - 1):
            start, end = pivots[k], pivots[k + 1]
            magnitude = end.price - start.price
            direction = "up" if magnitude > 0 else "down"
            start_atr = atr[start.idx]
            if not np.isfinite(start_atr) or start_atr <= 0:
                start_atr = float(np.nanmean(atr)) if np.isfinite(np.nanmean(atr)) else 100.0
            tol = zone_atr_multiple * start_atr

            zones = {}
            if prev is not None:
                zones["prev_day_high"] = abs(start.price - prev["day_high"])
                zones["prev_day_low"] = abs(start.price - prev["day_low"])
                zones["prev_day_close"] = abs(start.price - prev["day_close"])
            for lvl in recent_levels:
                zones.setdefault("recent_swing", 1e18)
                zones["recent_swing"] = min(zones["recent_swing"], abs(start.price - lvl))
            near_zone = None
            near_zone_dist_atr = None
            for name, dist in zones.items():
                if dist <= tol and (near_zone_dist_atr is None or dist < near_zone_dist_atr):
                    near_zone = name
                    near_zone_dist_atr = dist / start_atr

            # pre-move candle cluster: cluster_lookback bars ending at (and including) the start pivot
            lo_idx = max(0, start.idx - cluster_lookback + 1)
            seg_o = o[lo_idx : start.idx + 1]
            seg_h = h[lo_idx : start.idx + 1]
            seg_l = lo[lo_idx : start.idx + 1]
            seg_c = c[lo_idx : start.idx + 1]
            if len(seg_c) >= 2:
                net_move = float(seg_c[-1] - seg_c[0])
                gross_range = float(np.sum(seg_h - seg_l))
                efficiency = abs(net_move) / gross_range if gross_range > 0 else 0.0
                up_count = int(np.sum(seg_c > seg_o))
                down_count = int(np.sum(seg_c < seg_o))
            else:
                net_move = 0.0
                efficiency = 0.0
                up_count = down_count = 0

            pre_move_aligned = (net_move > 0 and direction == "up") or (
                net_move < 0 and direction == "down"
            )

            rows.append(
                {
                    "date": d,
                    "start_time": str(start.timestamp),
                    "end_time": str(end.timestamp),
                    "direction": direction,
                    "magnitude_points": abs(magnitude),
                    "duration_bars": end.idx - start.idx,
                    "start_price": start.price,
                    "end_price": end.price,
                    "start_atr14": float(start_atr),
                    "near_zone": near_zone,
                    "near_zone_dist_atr": near_zone_dist_atr,
                    "pre_move_net_move": net_move,
                    "pre_move_efficiency": efficiency,
                    "pre_move_up_count": up_count,
                    "pre_move_down_count": down_count,
                    "pre_move_aligned_with_move": pre_move_aligned,
                    "pre_move_compression": efficiency < 0.4,
                    "minutes_from_open": int((start.timestamp - ts[0]) / np.timedelta64(1, "m")),
                }
            )

    return pl.DataFrame(rows) if rows else pl.DataFrame()
