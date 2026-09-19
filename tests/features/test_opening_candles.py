from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl

from src.features.opening_candles import add_opening_candle_features

_TZ = ZoneInfo("Asia/Kolkata")


def _synthetic_session(date, base=100.0):
    # 10 bars @ 5m: candle1 (bars 0-2) trends up, candle2 (bars 3-5) trends up too
    rows = []
    prices = [
        base,
        base + 5,
        base + 10,
        base + 12,
        base + 18,
        base + 24,
        base + 20,
        base + 22,
        base + 25,
        base + 28,
    ]
    for i, p in enumerate(prices):
        minute = 20 + 5 * i
        hour = 9 + minute // 60
        minute = minute % 60
        ts = datetime(2024, 1, date, hour, minute, tzinfo=_TZ)
        rows.append({"timestamp": ts, "open": p, "high": p + 2, "low": p - 2, "close": p + 1})
    return rows


def test_opening_candle_features_null_before_second_candle_closes():
    rows = []
    for d in range(1, 25):
        rows.extend(_synthetic_session(d, base=100.0 + d))
    df = pl.DataFrame(rows)
    out = add_opening_candle_features(df)

    day1 = out.filter(pl.col("timestamp").dt.date() == out["timestamp"][0].date()).sort("timestamp")
    # first 5 bars (session_bar_index 0-4) precede the second candle's close
    assert day1["f_c1_direction"][:5].null_count() == 5
    # bar index 5 (09:45, the close of candle2) onward should be populated
    assert day1["f_c1_direction"][5] is not None
    assert day1["f_c1_c2_aligned"][5] in (0, 1)


def test_opening_candle_direction_matches_synthetic_uptrend():
    rows = []
    for d in range(1, 25):
        rows.extend(_synthetic_session(d, base=100.0 + d))
    df = pl.DataFrame(rows)
    out = add_opening_candle_features(df)
    last_day = out.filter(pl.col("timestamp").dt.date() == out["timestamp"][-1].date()).sort(
        "timestamp"
    )
    row = last_day.filter(pl.col("f_session_bar_index") == 5)
    assert row["f_c1_direction"][0] == 1
    assert row["f_c2_direction"][0] == 1
    assert row["f_c1_c2_aligned"][0] == 1
