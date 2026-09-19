"""Causal features describing the first two 15-min candles of the session
(each = 3 consecutive 5m bars: 09:20-09:35 and 09:35-09:50). All values are
ATR-normalized or ratio-based (never raw price/point levels) since Bank
Nifty's price drifted ~36,000->57,000 over the 5-year history. Every
feature is null until the second candle has actually closed (session bar
index >= 5, i.e. from 09:50 onward) -- before that a real-time system
would not yet know these values either.
"""

from __future__ import annotations

import polars as pl


def add_opening_candle_features(df: pl.DataFrame) -> pl.DataFrame:
    eps = 1e-9
    out = df.sort("timestamp")
    if "f_session_bar_index" not in out.columns:
        from .time_features import add_time_features

        out = add_time_features(out)
    needs_atr = "f_atr_14" not in out.columns
    if needs_atr:
        from .volatility import add_volatility_features

        out = add_volatility_features(out)
        if "_session_date" not in out.columns:
            out = out.with_columns(pl.col("timestamp").dt.date().alias("_session_date"))
    else:
        out = out.with_columns(pl.col("timestamp").dt.date().alias("_session_date"))

    def _candle_stats(bar_lo: int, bar_hi: int, prefix: str) -> pl.DataFrame:
        window = out.filter(
            (pl.col("f_session_bar_index") >= bar_lo) & (pl.col("f_session_bar_index") <= bar_hi)
        )
        return window.group_by("_session_date").agg(
            pl.col("open").first().alias(f"_{prefix}_open"),
            pl.col("high").max().alias(f"_{prefix}_high"),
            pl.col("low").min().alias(f"_{prefix}_low"),
            pl.col("close").last().alias(f"_{prefix}_close"),
        )

    c1 = _candle_stats(0, 2, "c1")
    c2 = _candle_stats(3, 5, "c2")
    prev_close = out.group_by("_session_date").agg(pl.col("open").first().alias("_day_open"))
    out = out.join(c1, on="_session_date", how="left").join(c2, on="_session_date", how="left")

    ready = pl.col("f_session_bar_index") >= 5

    def _candle_feats(prefix: str) -> list[pl.Expr]:
        o, h, low, c = (
            pl.col(f"_{prefix}_open"),
            pl.col(f"_{prefix}_high"),
            pl.col(f"_{prefix}_low"),
            pl.col(f"_{prefix}_close"),
        )
        rng = h - low
        body = (c - o).abs()
        upper_wick = h - pl.max_horizontal(o, c)
        lower_wick = pl.min_horizontal(o, c) - low
        atr = pl.col("f_atr_14") + eps
        return [
            pl.when(ready).then(rng / atr).otherwise(None).alias(f"f_{prefix}_range_atr"),
            pl.when(ready).then(body / (rng + eps)).otherwise(None).alias(f"f_{prefix}_body_ratio"),
            pl.when(ready)
            .then(upper_wick / (rng + eps))
            .otherwise(None)
            .alias(f"f_{prefix}_upper_wick_ratio"),
            pl.when(ready)
            .then(lower_wick / (rng + eps))
            .otherwise(None)
            .alias(f"f_{prefix}_lower_wick_ratio"),
            pl.when(ready)
            .then(pl.when(c > o).then(1).when(c < o).then(-1).otherwise(0))
            .otherwise(None)
            .alias(f"f_{prefix}_direction"),
        ]

    out = out.with_columns(*_candle_feats("c1"), *_candle_feats("c2"))
    out = out.with_columns(
        pl.when(ready)
        .then(
            (pl.col("_c1_open") - pl.col("close").shift(1).over("_session_date"))
            / (pl.col("f_atr_14") + eps)
        )
        .otherwise(None)
        .alias("f_opening_gap_atr"),
        pl.when(ready)
        .then((pl.col("f_c1_direction") == pl.col("f_c2_direction")).cast(pl.Int8))
        .otherwise(None)
        .alias("f_c1_c2_aligned"),
        pl.when(ready)
        .then(((pl.col("_c2_high") - pl.col("_c1_low")) + (pl.col("_c1_high") - pl.col("_c2_low"))))
        .otherwise(None)
        .alias("_combined_range_tmp"),
    )
    out = out.with_columns(
        pl.when(ready)
        .then(
            (pl.max_horizontal("_c1_high", "_c2_high") - pl.min_horizontal("_c1_low", "_c2_low"))
            / (pl.col("f_atr_14") + eps)
        )
        .otherwise(None)
        .alias("f_c1c2_combined_range_atr")
    )
    drop_cols = [
        "_session_date",
        "_c1_open",
        "_c1_high",
        "_c1_low",
        "_c1_close",
        "_c2_open",
        "_c2_high",
        "_c2_low",
        "_c2_close",
        "_combined_range_tmp",
    ]
    return out.drop([c for c in drop_cols if c in out.columns])
