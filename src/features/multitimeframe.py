from __future__ import annotations

import polars as pl


def _causal_aggregate(df: pl.DataFrame, minutes: int) -> pl.DataFrame:
    """Aggregate already-completed 5m closes into causal higher-timeframe state."""
    if minutes % 5:
        raise ValueError("minutes must be a multiple of 5")
    bars=minutes//5
    work=df.select(["timestamp","high","low","close","volume"]).sort("timestamp")
    # Rolling windows never include future rows. Values are joined back only to
    # the current 5m timestamp, avoiding forward-looking resampling.
    return work.with_columns(
        pl.col("close").rolling_mean(bars).alias(f"_ma_{minutes}"),
        pl.col("high").rolling_max(bars).alias(f"_high_{minutes}"),
        pl.col("low").rolling_min(bars).alias(f"_low_{minutes}"),
        pl.col("volume").rolling_mean(bars).alias(f"_volume_{minutes}"),
        pl.col("close").pct_change(bars).alias(f"_return_{minutes}"),
    ).select(
        "timestamp",
        f"_ma_{minutes}",f"_high_{minutes}",f"_low_{minutes}",
        f"_volume_{minutes}",f"_return_{minutes}",
    )


def add_multi_timeframe_features(
    df: pl.DataFrame,
    timeframes: tuple[int,...]=(15,30,60),
) -> pl.DataFrame:
    out=df.sort("timestamp")
    for minutes in timeframes:
        context=_causal_aggregate(out,minutes)
        out=out.join(context,on="timestamp",how="left").with_columns(
            (
                pl.col("close")-pl.col(f"_ma_{minutes}")
            ).alias(f"f_mtf_{minutes}_distance_ma"),
            (
                (pl.col("close")-pl.col(f"_low_{minutes}"))
                /(pl.col(f"_high_{minutes}")-pl.col(f"_low_{minutes}")+1e-9)
            ).alias(f"f_mtf_{minutes}_range_position"),
        ).drop([
            f"_ma_{minutes}",f"_high_{minutes}",f"_low_{minutes}",
            f"_volume_{minutes}"
        ]).rename({
            f"_return_{minutes}":f"f_mtf_{minutes}_return"
        })
    return out
