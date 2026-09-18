from __future__ import annotations

import polars as pl


def add_multi_timeframe_features(
    df:pl.DataFrame,
    timeframes:tuple[int,...]=(15,30,60),
)->pl.DataFrame:
    """Add causal higher-timeframe context without cross-session windows."""
    out=df.sort("timestamp").with_columns(
        pl.col("timestamp").dt.date().alias("_session_date")
    )
    for minutes in timeframes:
        bars=max(1,minutes//5)
        out=out.with_columns(
            pl.col("close").rolling_mean(bars).over("_session_date").alias(f"_ma_{minutes}"),
            pl.col("high").rolling_max(bars).over("_session_date").alias(f"_high_{minutes}"),
            pl.col("low").rolling_min(bars).over("_session_date").alias(f"_low_{minutes}"),
            pl.col("close").pct_change(bars).over("_session_date").alias(f"_ret_{minutes}"),
        ).with_columns(
            (pl.col("close")-pl.col(f"_ma_{minutes}")).alias(f"f_mtf_{minutes}_distance_ma"),
            (
                (pl.col("close")-pl.col(f"_low_{minutes}"))
                /(pl.col(f"_high_{minutes}")-pl.col(f"_low_{minutes}")+1e-9)
            ).alias(f"f_mtf_{minutes}_range_position"),
            pl.col(f"_ret_{minutes}").alias(f"f_mtf_{minutes}_return"),
        ).drop([f"_ma_{minutes}",f"_high_{minutes}",f"_low_{minutes}",f"_ret_{minutes}"])
    return out.drop("_session_date")
