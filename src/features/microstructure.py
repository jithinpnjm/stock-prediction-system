from __future__ import annotations

import polars as pl


def add_1m_inside_5m_features(
    df_1m: pl.DataFrame,
    df_5m: pl.DataFrame,
) -> pl.DataFrame:
    one = df_1m.sort("timestamp").with_columns(
        pl.col("timestamp").dt.date().alias("_date"),
        (
            pl.col("timestamp").dt.hour() * 60 + pl.col("timestamp").dt.minute()
        ).alias("_minute"),
        (pl.col("close") > pl.col("open")).cast(pl.Int8).alias("_up"),
        (pl.col("close") < pl.col("open")).cast(pl.Int8).alias("_down"),
        (pl.col("high") - pl.col("low")).alias("_range"),
    )
    five = df_5m.sort("timestamp").with_columns(
        pl.col("timestamp").dt.date().alias("_date"),
        (
            pl.col("timestamp").dt.hour() * 60 + pl.col("timestamp").dt.minute()
        ).alias("_close_minute"),
    )
    summary = (
        five.join(
            one.select(
                [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "_date",
                    "_minute",
                    "_up",
                    "_down",
                    "_range",
                ]
            ),
            on="_date",
            how="left",
        )
        .filter(
            (pl.col("_minute") <= pl.col("_close_minute"))
            & (pl.col("_minute") > pl.col("_close_minute") - 5)
        )
        .group_by("timestamp")
        .agg(
            pl.col("high").max().alias("f_1m_path_high"),
            pl.col("low").min().alias("f_1m_path_low"),
            pl.col("open").first().alias("_first_1m_open"),
            pl.col("close").last().alias("_last_1m_close"),
            pl.col("volume").sum().alias("f_1m_path_volume"),
            pl.col("_range").sum().alias("f_1m_range_sum"),
            pl.col("_range").mean().alias("f_1m_range_mean"),
            pl.col("_range").std().alias("f_1m_range_std"),
            pl.col("_up").sum().alias("f_1m_up_count"),
            pl.col("_down").sum().alias("f_1m_down_count"),
            pl.len().alias("f_1m_count"),
        )
        .with_columns(
            (
                pl.col("_last_1m_close") - pl.col("_first_1m_open")
            ).alias("f_1m_path_net_move"),
            (
                pl.col("f_1m_path_high") - pl.col("f_1m_path_low")
            ).alias("f_1m_path_range"),
            (
                pl.col("f_1m_path_high") - pl.col("_last_1m_close")
            ).alias("f_1m_path_high_rejection"),
            (
                pl.col("_last_1m_close") - pl.col("f_1m_path_low")
            ).alias("f_1m_path_low_rebound"),
            (
                pl.col("f_1m_up_count") / (pl.col("f_1m_count") + 1e-9)
            ).alias("f_1m_up_ratio"),
        )
        .drop(["_first_1m_open", "_last_1m_close"])
    )
    return five.join(summary, on="timestamp", how="left").drop(
        ["_date", "_close_minute"]
    )
