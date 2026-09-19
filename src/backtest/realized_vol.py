"""Annualized realized-volatility proxy for implied vol, used by the
options P&L backtest since we don't have historical Bank Nifty IV data.
This is a standard substitute (real IV usually runs somewhat above
realized vol -- the "variance risk premium" -- so option P&L simulated
this way is, if anything, a bit optimistic for a buyer since it
understates the premium actually paid)."""

from __future__ import annotations

import polars as pl

TRADING_DAYS_PER_YEAR = 252


def add_realized_vol(df_daily: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """df_daily: one row per session with a `day_close` column, sorted by date."""
    return (
        df_daily.with_columns(
            (pl.col("day_close").log() - pl.col("day_close").log().shift(1)).alias("_log_ret")
        )
        .with_columns(
            (
                pl.col("_log_ret").rolling_std(window_size=window) * (TRADING_DAYS_PER_YEAR**0.5)
            ).alias("realized_vol")
        )
        .drop("_log_ret")
    )
