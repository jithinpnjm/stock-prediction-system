from __future__ import annotations

import polars as pl


def add_market_structure_features(df: pl.DataFrame) -> pl.DataFrame:
    out = df.sort("timestamp")
    if "f_swing_high_candidate" not in out.columns:
        from .swings import add_swing_features
        out = add_swing_features(out)
    prior_high = pl.col("high").shift(1).rolling_max(12)
    prior_low = pl.col("low").shift(1).rolling_min(12)
    prior_range_high = pl.col("high").shift(1).rolling_max(48)
    prior_range_low = pl.col("low").shift(1).rolling_min(48)
    return out.with_columns(
        (pl.col("high") > prior_high).cast(pl.Int8).alias("f_break_of_structure_up"),
        (pl.col("low") < prior_low).cast(pl.Int8).alias("f_break_of_structure_down"),
        (pl.col("close") > prior_range_high).cast(pl.Int8).alias("f_range_breakout_up"),
        (pl.col("close") < prior_range_low).cast(pl.Int8).alias("f_range_breakout_down"),
        (
            (pl.col("close") - prior_range_low)
            / (prior_range_high - prior_range_low + 1e-9)
        ).alias("f_range_position"),
        (
            pl.col("close").rolling_mean(12)
            - pl.col("close").rolling_mean(48)
        ).alias("f_trend_context"),
        (
            pl.col("high").rolling_max(12) - pl.col("low").rolling_min(12)
        ).alias("f_short_structure_range"),
    )
