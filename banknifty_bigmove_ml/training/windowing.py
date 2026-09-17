"""
windowing.py
-------------
Turns the Phase 1 full parquet (every row, label=NaN for context-only rows —
see data_pipeline/build_labeled_dataset.py) into fixed-length raw-OHLCV
sequence windows for a sequence model — optionally across multiple
timeframes (e.g. 1-min + 5-min) fused as separate encoder branches.

No hand-crafted indicators: each window's open/high/low/close are simply
expressed as a return relative to that window's own last close (the "now"
price) — this is normalization, not feature engineering, and is required
because BankNifty's absolute price level drifted from ~35,000 to ~55,000+
over the 5-year dataset, so raw price values aren't stationary/comparable
across time on their own.

Built once over the WHOLE dataset (context rows included, so windows can
legitimately span a session boundary — that's real market information, e.g.
an overnight gap, not leakage). Fold slicing then happens by date on the
resulting arrays, reusing splits.py's fold boundaries.

Per-candle channels (all deterministic, lossless functions of OHLC — not
hand-picked indicators, just a more learnable representation of the same
raw price action; the model still decides what matters, the walk-forward
gate still validates it):
  0 open_rel        open  / anchor_close - 1
  1 high_rel        high  / anchor_close - 1
  2 low_rel         low   / anchor_close - 1
  3 close_rel       close / anchor_close - 1
  4 body            (close - open) / anchor_close        (signed size, relative to "now")
  5 upper_wick      (high - max(open,close)) / anchor_close
  6 lower_wick      (min(open,close) - low) / anchor_close
  7 range           (high - low) / anchor_close
  8 body_to_range   |close-open| / (high-low), scale-invariant, 0 if high==low
  9 color           sign(close - open): -1 red, 0 doji, +1 green

Multi-timeframe (5-min etc.): derived by resampling the SAME 1-min data
(open=first, high=max, low=min, close=last per bucket) rather than fetching
a second data source, so there's exactly one source of truth. A higher-
timeframe bar is only usable for a target candle at time t if the bar is
FULLY CLOSED as of t (bar_end <= t) — the in-progress bucket containing t
itself is never visible, which is what prevents lookahead leakage through
the coarser timeframe.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

N_FEATURES = 10


def _ohlc_shape_channels(o, h, l, c, anchor):
    """o,h,l,c: (n, lookback) raw price arrays. anchor: (n,) raw price to
    normalize against (the target candle's own close). Returns (n, lookback, N_FEATURES)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        o_rel = o / anchor[:, None] - 1.0
        h_rel = h / anchor[:, None] - 1.0
        l_rel = l / anchor[:, None] - 1.0
        c_rel = c / anchor[:, None] - 1.0
        body = (c - o) / anchor[:, None]
        upper_wick = (h - np.maximum(o, c)) / anchor[:, None]
        lower_wick = (np.minimum(o, c) - l) / anchor[:, None]
        rng = (h - l) / anchor[:, None]

    raw_range = h - l
    body_to_range = np.divide(
        np.abs(c - o), raw_range,
        out=np.zeros_like(raw_range, dtype=np.float64), where=raw_range > 1e-9,
    )
    color = np.sign(c - o)

    X = np.stack(
        [o_rel, h_rel, l_rel, c_rel, body, upper_wick, lower_wick, rng, body_to_range, color],
        axis=-1,
    ).astype(np.float32)
    assert X.shape[-1] == N_FEATURES
    return X


@dataclass
class WindowedDataset:
    X: np.ndarray          # (n_samples, lookback, N_FEATURES) float32
    y: np.ndarray           # (n_samples,) int64 — 0/1 label
    dates: np.ndarray       # (n_samples,) datetime64[D] — target candle's session date
    datetimes: np.ndarray   # (n_samples,) datetime64[ns] — target candle's exact timestamp
    anchors: np.ndarray = field(default=None)  # (n_samples,) raw close price of the target candle


def build_windows(df: pd.DataFrame, lookback: int) -> WindowedDataset:
    df = df.sort_values("datetime").reset_index(drop=True)
    n = len(df)
    if n <= lookback:
        raise ValueError(f"Not enough rows ({n}) for lookback={lookback}")

    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    labels = df["label"].values  # float, NaN for context-only rows
    dates = pd.to_datetime(df["date"]).values
    datetimes = df["datetime"].values

    o_w = sliding_window_view(opens, lookback)   # window i covers rows [i, i+lookback-1]
    h_w = sliding_window_view(highs, lookback)
    l_w = sliding_window_view(lows, lookback)
    c_w = sliding_window_view(closes, lookback)

    anchor = c_w[:, -1]  # close of the LAST row in each window == the target candle's close
    X_all = _ohlc_shape_channels(o_w, h_w, l_w, c_w, anchor)

    # window i's target row t = i + lookback - 1
    labels_for_windows = labels[lookback - 1:]
    dates_for_windows = dates[lookback - 1:]
    datetimes_for_windows = datetimes[lookback - 1:]
    anchor_for_windows = anchor

    mask = ~np.isnan(labels_for_windows)
    X = X_all[mask]
    y = labels_for_windows[mask].astype(np.int64)
    dates_out = dates_for_windows[mask]
    datetimes_out = datetimes_for_windows[mask]
    anchors_out = anchor_for_windows[mask]

    if not np.isfinite(X).all():
        raise ValueError(
            "Non-finite values in windowed features — check for zero/negative closes "
            "in the source data (division by anchor close)."
        )

    return WindowedDataset(X=X, y=y, dates=dates_out, datetimes=datetimes_out, anchors=anchors_out)


def slice_by_date(ds: WindowedDataset, start, end) -> WindowedDataset:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    mask = (ds.dates >= np.datetime64(start)) & (ds.dates <= np.datetime64(end))
    anchors = ds.anchors[mask] if ds.anchors is not None else None
    return WindowedDataset(X=ds.X[mask], y=ds.y[mask], dates=ds.dates[mask], datetimes=ds.datetimes[mask], anchors=anchors)


# ============================================================================
# Multi-timeframe support
# ============================================================================
def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Derive higher-timeframe bars from the 1-min data (single source of
    truth). Bucket boundaries align to wall-clock time (e.g. '5min' buckets
    at 09:15-09:19, 09:20-09:24, ...) which happens to line up cleanly since
    BankNifty's 09:15 session open is itself a multiple of 5 minutes."""
    d = df.set_index("datetime")[["open", "high", "low", "close"]].sort_index()
    agg = d.resample(rule, label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
    )
    agg = agg.dropna(how="any")  # buckets with no trading (overnight/weekend) produce empty rows
    agg = agg.reset_index().rename(columns={"datetime": "bar_start"})
    agg["bar_end"] = agg["bar_start"] + pd.to_timedelta(rule)
    return agg


@dataclass
class MultiTimeframeDataset:
    X_by_tf: dict           # {"1min": (n, lookback_1m, N_FEATURES), "5min": (n, lookback_htf, N_FEATURES), ...}
    y: np.ndarray
    dates: np.ndarray
    datetimes: np.ndarray


def build_multi_timeframe_windows(
    df: pd.DataFrame,
    base_lookback: int,
    htf_specs: dict,   # {"5min": 24, "15min": 16, ...} — rule -> lookback bars
) -> MultiTimeframeDataset:
    """base timeframe is always the native 1-min data. Each entry in
    htf_specs adds a branch built from FULLY CLOSED higher-timeframe bars
    only (no lookahead through the coarser timeframe)."""
    ds_1m = build_windows(df, lookback=base_lookback)

    combined_mask = np.ones(len(ds_1m.y), dtype=bool)
    htf_arrays = {}

    for rule, htf_lookback in htf_specs.items():
        htf = resample_ohlc(df, rule)
        bar_end = htf["bar_end"].values
        o = htf["open"].values.astype(np.float64)
        h = htf["high"].values.astype(np.float64)
        l = htf["low"].values.astype(np.float64)
        c = htf["close"].values.astype(np.float64)

        if len(htf) <= htf_lookback:
            raise ValueError(f"Not enough {rule} bars ({len(htf)}) for htf_lookback={htf_lookback}")

        # index of the last FULLY CLOSED htf bar as of each 1-min target's timestamp
        last_closed_idx = np.searchsorted(bar_end, ds_1m.datetimes, side="right") - 1
        valid = last_closed_idx >= (htf_lookback - 1)

        o_w = sliding_window_view(o, htf_lookback)  # window i ends at htf bar index i+htf_lookback-1
        h_w = sliding_window_view(h, htf_lookback)
        l_w = sliding_window_view(l, htf_lookback)
        c_w = sliding_window_view(c, htf_lookback)

        gather_i = np.clip(last_closed_idx - htf_lookback + 1, 0, o_w.shape[0] - 1)
        o_sel, h_sel, l_sel, c_sel = o_w[gather_i], h_w[gather_i], l_w[gather_i], c_w[gather_i]

        # anchor to the 1-min target's own close, matching the base branch's scale
        X_htf = _ohlc_shape_channels(o_sel, h_sel, l_sel, c_sel, ds_1m.anchors)

        htf_arrays[rule] = X_htf
        combined_mask &= valid

    X_by_tf = {"1min": ds_1m.X[combined_mask]}
    for rule, X_htf in htf_arrays.items():
        X_by_tf[rule] = X_htf[combined_mask]

    return MultiTimeframeDataset(
        X_by_tf=X_by_tf,
        y=ds_1m.y[combined_mask],
        dates=ds_1m.dates[combined_mask],
        datetimes=ds_1m.datetimes[combined_mask],
    )


def slice_mtf_by_date(ds: MultiTimeframeDataset, start, end) -> MultiTimeframeDataset:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    mask = (ds.dates >= np.datetime64(start)) & (ds.dates <= np.datetime64(end))
    return MultiTimeframeDataset(
        X_by_tf={tf: X[mask] for tf, X in ds.X_by_tf.items()},
        y=ds.y[mask], dates=ds.dates[mask], datetimes=ds.datetimes[mask],
    )


# ============================================================================
# Directional (triple-barrier) windowing — locked 2026-09-13, replaces the
# earlier ±0.3%-any-direction label as the real training target. Two dual
# targets (long_wins, short_wins) per candle instead of one. Adds objective
# previous-day reference levels as extra channels — NOT pattern detection,
# just numbers the model's lookback window otherwise can't see at all
# (yesterday's high/low is outside a 60-120 minute window by construction).
# Kept as new functions alongside the original ones so nothing about the
# already-registered v1/v2 models or their reproducibility is disturbed.
# ============================================================================
N_EXTRA_CONTEXT_FEATURES = 3   # prev_day_high_rel, prev_day_low_rel, prev_day_close_rel
N_FEATURES_DIRECTIONAL = N_FEATURES + N_EXTRA_CONTEXT_FEATURES


def compute_prev_day_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Returns df with three added columns: prev_day_high, prev_day_low,
    prev_day_close — the PRIOR trading session's levels, merged onto every
    row of that day. The first day in the dataset has no prior session, so
    its rows get NaN (excluded downstream, same as any other undefined row)."""
    df = df.copy()
    daily = df.groupby("date").agg(
        day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last"),
    ).sort_index()
    daily["prev_day_high"] = daily["day_high"].shift(1)
    daily["prev_day_low"] = daily["day_low"].shift(1)
    daily["prev_day_close"] = daily["day_close"].shift(1)
    return df.merge(
        daily[["prev_day_high", "prev_day_low", "prev_day_close"]],
        left_on="date", right_index=True, how="left",
    )


def _append_context_channels(X: np.ndarray, anchor: np.ndarray,
                              prev_high: np.ndarray, prev_low: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
    """X: (n, lookback, N_FEATURES). Broadcasts 3 per-sample scalars (constant
    across the lookback/time dimension — they don't change minute to minute)
    into 3 additional channels, anchored the same way as everything else."""
    lookback = X.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ph_rel = (prev_high / anchor - 1.0)[:, None].repeat(lookback, axis=1)
        pl_rel = (prev_low / anchor - 1.0)[:, None].repeat(lookback, axis=1)
        pc_rel = (prev_close / anchor - 1.0)[:, None].repeat(lookback, axis=1)
    extra = np.stack([ph_rel, pl_rel, pc_rel], axis=-1).astype(np.float32)
    return np.concatenate([X, extra], axis=-1)


@dataclass
class DirectionalWindowedDataset:
    X: np.ndarray          # (n_samples, lookback, N_FEATURES_DIRECTIONAL) float32
    y: np.ndarray           # (n_samples, 2) float32 — [long_wins, short_wins], both 0/1
    dates: np.ndarray
    datetimes: np.ndarray
    anchors: np.ndarray = field(default=None)


def build_windows_directional(df: pd.DataFrame, lookback: int) -> DirectionalWindowedDataset:
    """df must already have long_wins/short_wins columns (see
    data_pipeline/triple_barrier_label.py) and a `date` column."""
    df = compute_prev_day_levels(df)
    df = df.sort_values("datetime").reset_index(drop=True)
    n = len(df)
    if n <= lookback:
        raise ValueError(f"Not enough rows ({n}) for lookback={lookback}")

    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    long_wins = df["long_wins"].values
    short_wins = df["short_wins"].values
    prev_high = df["prev_day_high"].values.astype(np.float64)
    prev_low = df["prev_day_low"].values.astype(np.float64)
    prev_close = df["prev_day_close"].values.astype(np.float64)
    dates = pd.to_datetime(df["date"]).values
    datetimes = df["datetime"].values

    o_w = sliding_window_view(opens, lookback)
    h_w = sliding_window_view(highs, lookback)
    l_w = sliding_window_view(lows, lookback)
    c_w = sliding_window_view(closes, lookback)
    anchor = c_w[:, -1]
    X_base = _ohlc_shape_channels(o_w, h_w, l_w, c_w, anchor)

    # window i's target row t = i + lookback - 1 — same alignment as build_windows
    long_for_windows = long_wins[lookback - 1:]
    short_for_windows = short_wins[lookback - 1:]
    prev_high_for_windows = prev_high[lookback - 1:]
    prev_low_for_windows = prev_low[lookback - 1:]
    prev_close_for_windows = prev_close[lookback - 1:]
    dates_for_windows = dates[lookback - 1:]
    datetimes_for_windows = datetimes[lookback - 1:]

    X_all = _append_context_channels(X_base, anchor, prev_high_for_windows, prev_low_for_windows, prev_close_for_windows)

    # valid only where both targets are defined AND prev-day levels exist (first day excluded)
    mask = (~np.isnan(long_for_windows)) & (~np.isnan(short_for_windows)) & (~np.isnan(prev_high_for_windows))
    X = X_all[mask]
    y = np.stack([long_for_windows[mask], short_for_windows[mask]], axis=-1).astype(np.float32)
    dates_out = dates_for_windows[mask]
    datetimes_out = datetimes_for_windows[mask]
    anchors_out = anchor[mask]

    if not np.isfinite(X).all():
        raise ValueError("Non-finite values in directional windowed features.")

    return DirectionalWindowedDataset(X=X, y=y, dates=dates_out, datetimes=datetimes_out, anchors=anchors_out)


def slice_directional_by_date(ds: DirectionalWindowedDataset, start, end) -> DirectionalWindowedDataset:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    mask = (ds.dates >= np.datetime64(start)) & (ds.dates <= np.datetime64(end))
    anchors = ds.anchors[mask] if ds.anchors is not None else None
    return DirectionalWindowedDataset(X=ds.X[mask], y=ds.y[mask], dates=ds.dates[mask],
                                       datetimes=ds.datetimes[mask], anchors=anchors)


@dataclass
class MultiTimeframeDirectionalDataset:
    X_by_tf: dict
    y: np.ndarray           # (n, 2) — [long_wins, short_wins]
    dates: np.ndarray
    datetimes: np.ndarray


def build_multi_timeframe_windows_directional(
    df: pd.DataFrame, base_lookback: int, htf_specs: dict,
) -> MultiTimeframeDirectionalDataset:
    ds_1m = build_windows_directional(df, lookback=base_lookback)

    combined_mask = np.ones(len(ds_1m.y), dtype=bool)
    htf_arrays = {}
    for rule, htf_lookback in htf_specs.items():
        htf = resample_ohlc(df, rule)
        bar_end = htf["bar_end"].values
        o = htf["open"].values.astype(np.float64)
        h = htf["high"].values.astype(np.float64)
        l = htf["low"].values.astype(np.float64)
        c = htf["close"].values.astype(np.float64)
        if len(htf) <= htf_lookback:
            raise ValueError(f"Not enough {rule} bars ({len(htf)}) for htf_lookback={htf_lookback}")

        last_closed_idx = np.searchsorted(bar_end, ds_1m.datetimes, side="right") - 1
        valid = last_closed_idx >= (htf_lookback - 1)

        o_w = sliding_window_view(o, htf_lookback)
        h_w = sliding_window_view(h, htf_lookback)
        l_w = sliding_window_view(l, htf_lookback)
        c_w = sliding_window_view(c, htf_lookback)

        gather_i = np.clip(last_closed_idx - htf_lookback + 1, 0, o_w.shape[0] - 1)
        o_sel, h_sel, l_sel, c_sel = o_w[gather_i], h_w[gather_i], l_w[gather_i], c_w[gather_i]
        X_htf = _ohlc_shape_channels(o_sel, h_sel, l_sel, c_sel, ds_1m.anchors)

        htf_arrays[rule] = X_htf
        combined_mask &= valid

    X_by_tf = {"1min": ds_1m.X[combined_mask]}
    for rule, X_htf in htf_arrays.items():
        X_by_tf[rule] = X_htf[combined_mask]

    return MultiTimeframeDirectionalDataset(
        X_by_tf=X_by_tf, y=ds_1m.y[combined_mask],
        dates=ds_1m.dates[combined_mask], datetimes=ds_1m.datetimes[combined_mask],
    )


def slice_mtf_directional_by_date(ds: MultiTimeframeDirectionalDataset, start, end) -> MultiTimeframeDirectionalDataset:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    mask = (ds.dates >= np.datetime64(start)) & (ds.dates <= np.datetime64(end))
    return MultiTimeframeDirectionalDataset(
        X_by_tf={tf: X[mask] for tf, X in ds.X_by_tf.items()},
        y=ds.y[mask], dates=ds.dates[mask], datetimes=ds.datetimes[mask],
    )
