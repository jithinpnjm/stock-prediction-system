"""
triple_barrier_label.py
-------------------------
The REAL success criteria (locked 2026-09-13, replaces the earlier
±0.3%-any-direction label): for every candle before 3:00 PM, would a LONG
entered here hit +100 points before -50 points, and would a SHORT entered
here hit -100 points before +50 points, looking forward through the rest of
the trading session? Neither barrier hit before session close = loss/scratch.

Two independent binary outcomes per entry candle (long_wins, short_wins) —
not mutually exclusive by construction, though in practice usually at most
one resolves true. The eventual trading rule is: take whichever direction
has the higher predicted probability, only above a confidence threshold,
else skip.

Conservative tie-break: if a single forward candle's high AND low both
cross their respective barriers for a direction (both target and stop
touched in the same candle), the STOP is assumed to have hit first — never
assume the best case for an ambiguous candle.

No entries scored at/after 3:00 PM (context-only, same treatment as the
old label's session-tail rows) — matches the "no new trades after 3pm"
trading rule directly in the label itself, not bolted on afterward.
"""
from __future__ import annotations

import numpy as np


def triple_barrier_session(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    minute_of_day: np.ndarray,
    target_pts: float = 100.0, stop_pts: float = 50.0,
    entry_cutoff_minute: int = 15 * 60,  # 15:00 in minutes-since-midnight
):
    """All arrays are for ONE trading session (one calendar day), in
    chronological order. minute_of_day[i] = candle i's minutes-since-midnight
    (e.g. 9:15 -> 555), used only to apply the entry cutoff.

    Returns (long_wins, short_wins): float arrays, NaN where the candle
    isn't a valid entry (at/after the cutoff), else 1.0/0.0.
    """
    n = len(close)
    long_wins = np.full(n, np.nan)
    short_wins = np.full(n, np.nan)
    if n < 2:
        return long_wins, short_wins

    idx = np.arange(n)
    # (n, n) matrices: row = entry candle t, col = forward candle i
    high_grid = high[None, :]
    low_grid = low[None, :]
    close_col = close[:, None]

    forward_mask = idx[None, :] > idx[:, None]  # i > t only

    long_target_hit = (high_grid >= close_col + target_pts) & forward_mask
    long_stop_hit = (low_grid <= close_col - stop_pts) & forward_mask
    short_target_hit = (low_grid <= close_col - target_pts) & forward_mask
    short_stop_hit = (high_grid >= close_col + stop_pts) & forward_mask

    col_idx = np.broadcast_to(idx[None, :], (n, n))
    sentinel = n  # "never happens" — larger than any real index

    long_target_first = np.where(long_target_hit, col_idx, sentinel).min(axis=1)
    long_stop_first = np.where(long_stop_hit, col_idx, sentinel).min(axis=1)
    short_target_first = np.where(short_target_hit, col_idx, sentinel).min(axis=1)
    short_stop_first = np.where(short_stop_hit, col_idx, sentinel).min(axis=1)

    # Strict '<' — a tie (both hit on the same forward candle) resolves to the
    # stop side, since target_first == stop_first fails '<' and falls through
    # to the else-branch (loss), matching the conservative tie-break rule.
    long_wins_all = (long_target_first < sentinel) & (long_target_first < long_stop_first)
    short_wins_all = (short_target_first < sentinel) & (short_target_first < short_stop_first)

    valid_entry = minute_of_day < entry_cutoff_minute
    long_wins = np.where(valid_entry, long_wins_all.astype(float), np.nan)
    short_wins = np.where(valid_entry, short_wins_all.astype(float), np.nan)
    # last candle of the session can never resolve anything (no forward data at all)
    long_wins[-1] = np.nan
    short_wins[-1] = np.nan

    return long_wins, short_wins
