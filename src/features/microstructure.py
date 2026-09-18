from __future__ import annotations

import polars as pl


def add_1m_inside_5m_features(df_1m:pl.DataFrame,df_5m:pl.DataFrame)->pl.DataFrame:
    """Summarize the causal 1m path inside each completed 5m event."""
    one=df_1m.sort("timestamp")
    five=df_5m.sort("timestamp")
    local=one.with_columns(
        pl.col("timestamp").dt.date().alias("_date"),
        (
            pl.col("timestamp").dt.hour()*60+pl.col("timestamp").dt.minute()
        ).alias("_minute_of_day"),
    )
    # A 5m timestamp is its close time; map preceding 5 one-minute closes to it.
    result=five.with_columns(
        pl.col("timestamp").dt.date().alias("_date"),
        (
            pl.col("timestamp").dt.hour()*60+pl.col("timestamp").dt.minute()
        ).alias("_close_minute"),
    ).join(
        local.select(["timestamp","open","high","low","close","volume","_date","_minute_of_day"]),
        left_on="_date",right_on="_date",how="left"
    ).filter(
        (pl.col("_minute_of_day")<=pl.col("_close_minute"))
        &(pl.col("_minute_of_day")>pl.col("_close_minute")-5)
    ).group_by("timestamp").agg(
        pl.col("high").max().alias("f_1m_path_high"),
        pl.col("low").min().alias("f_1m_path_low"),
        pl.col("close").first().alias("_first_1m_close"),
        pl.col("close").last().alias("_last_1m_close"),
        pl.col("open").first().alias("_first_1m_open"),
        pl.col("volume").sum().alias("f_1m_path_volume"),
        pl.len().alias("f_1m_count"),
    ).with_columns(
        (
            pl.col("_last_1m_close")-pl.col("_first_1m_open")
        ).alias("f_1m_path_net_move"),
        (
            pl.col("f_1m_path_high")-pl.col("f_1m_path_low")
        ).alias("f_1m_path_range"),
        (
            pl.col("f_1m_path_high")-pl.col("_last_1m_close")
        ).alias("f_1m_path_high_rejection"),
        (
            pl.col("_last_1m_close")-pl.col("f_1m_path_low")
        ).alias("f_1m_path_low_rebound"),
    ).drop(["_first_1m_close","_last_1m_close","_first_1m_open"])
    return five.join(result,on="timestamp",how="left")
