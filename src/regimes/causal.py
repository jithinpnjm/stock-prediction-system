from __future__ import annotations

import numpy as np
import polars as pl


def add_causal_regime_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add regime features with all sequential state reset at session boundaries."""
    out = df.sort("timestamp").with_columns(
        pl.col("timestamp").dt.date().alias("_session_date")
    )
    ema_fast = pl.col("close").ewm_mean(span=8, adjust=False).over("_session_date")
    ema_slow = pl.col("close").ewm_mean(span=21, adjust=False).over("_session_date")
    trend = (ema_fast - ema_slow) / (pl.col("atr_14") + 1e-9)

    return out.with_columns(
        trend.alias("regime_trend_score"),
        pl.when(trend > 0.5).then(1).when(trend < -0.5).then(-1).otherwise(0)
        .alias("regime_trend_state"),
        (pl.col("atr_5") / (pl.col("atr_30") + 1e-9)).alias("regime_vol_ratio"),
        (pl.col("atr_5") < pl.col("atr_30")).cast(pl.Int8).alias("regime_compression"),
        (pl.col("atr_5") > pl.col("atr_30")).cast(pl.Int8).alias("regime_expansion"),
    ).drop("_session_date")


__all__ = ["add_causal_regime_features"]
