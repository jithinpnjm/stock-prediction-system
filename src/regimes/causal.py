from __future__ import annotations

import polars as pl


def add_causal_regime_features(df: pl.DataFrame) -> pl.DataFrame:
    ema_fast = pl.col("close").ewm_mean(span=8, adjust=False)
    ema_slow = pl.col("close").ewm_mean(span=21, adjust=False)
    trend = (ema_fast - ema_slow) / (pl.col("atr_14") + 1e-9)

    return df.with_columns(
        [
            trend.alias("regime_trend_score"),
            pl.when(trend > 0.5)
            .then(1)
            .when(trend < -0.5)
            .then(-1)
            .otherwise(0)
            .alias("regime_trend_state"),
            (
                pl.col("atr_5")
                / (pl.col("atr_30") + 1e-9)
            ).alias("regime_vol_ratio"),
            (
                pl.col("atr_5")
                < pl.col("atr_30")
            ).cast(pl.Int8).alias("regime_compression"),
            (
                pl.col("atr_5")
                > pl.col("atr_30")
            ).cast(pl.Int8).alias("regime_expansion"),
        ]
    )
