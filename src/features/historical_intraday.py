from __future__ import annotations

import polars as pl


def add_historical_intraday_context(
    df:pl.DataFrame,
    lookback_days:int=20,
)->pl.DataFrame:
    out=df.sort("timestamp")
    if "f_session_bar_index" not in out.columns:
        from .time_features import add_time_features
        out=add_time_features(out)

    out=out.with_columns(
        pl.col("timestamp").dt.date().alias("_date"),
        (
            pl.col("close")/pl.col("close").shift(1).over("_date")-1.0
        ).alias("_bar_return")
    )

    history=(
        out.select(["_date","f_session_bar_index","_bar_return"])
        .sort(["f_session_bar_index","_date"])
        .with_columns(
            pl.col("_bar_return")
            .shift(1)
            .rolling_mean(lookback_days)
            .over("f_session_bar_index")
            .alias("f_historical_slot_return_mean"),
            pl.col("_bar_return")
            .shift(1)
            .rolling_std(lookback_days)
            .over("f_session_bar_index")
            .alias("f_historical_slot_return_std"),
            (
                pl.col("_bar_return").shift(1).gt(0).cast(pl.Float64)
                .rolling_mean(lookback_days)
                .over("f_session_bar_index")
            ).alias("f_historical_slot_up_rate"),
        )
    )
    history=history.select(
        ["_date","f_session_bar_index",
         "f_historical_slot_return_mean",
         "f_historical_slot_return_std",
         "f_historical_slot_up_rate"]
    )
    return out.join(
        history,
        on=["_date","f_session_bar_index"],
        how="left",
    ).drop(["_date","_bar_return"])
