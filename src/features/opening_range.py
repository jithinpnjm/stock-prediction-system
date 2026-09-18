from __future__ import annotations

import polars as pl


def add_opening_range_features(
    df: pl.DataFrame,
    windows: tuple[int, ...] = (3, 6),
) -> pl.DataFrame:
    out = df.sort("timestamp")
    if "f_session_bar_index" not in out.columns:
        from .time_features import add_time_features
        out = add_time_features(out)
    out = out.with_columns(
        pl.col("timestamp").dt.date().alias("_session_date")
    )
    for n in windows:
        stats = (
            out.filter(pl.col("f_session_bar_index") < n)
            .group_by("_session_date")
            .agg(
                pl.col("high").max().alias(f"_or{n}_high"),
                pl.col("low").min().alias(f"_or{n}_low"),
                pl.col("open").first().alias(f"_or{n}_open"),
                pl.col("close").last().alias(f"_or{n}_close"),
            )
        )
        out = out.join(stats, on="_session_date", how="left").with_columns(
            pl.when(pl.col("f_session_bar_index") >= n - 1)
            .then(pl.col(f"_or{n}_high") - pl.col(f"_or{n}_low"))
            .otherwise(None).alias(f"f_or{n}_range"),
            pl.when(pl.col("f_session_bar_index") >= n - 1)
            .then(
                (pl.col("close") - pl.col(f"_or{n}_low"))
                / (pl.col(f"_or{n}_high") - pl.col(f"_or{n}_low") + 1e-9)
            )
            .otherwise(None).alias(f"f_or{n}_position"),
            pl.when(pl.col("f_session_bar_index") >= n - 1)
            .then(
                (pl.col("close") - pl.col(f"_or{n}_high"))
            )
            .otherwise(None).alias(f"f_distance_to_or{n}_high"),
            pl.when(pl.col("f_session_bar_index") >= n - 1)
            .then(
                (pl.col("close") - pl.col(f"_or{n}_low"))
            )
            .otherwise(None).alias(f"f_distance_to_or{n}_low"),
        ).drop([f"_or{n}_high", f"_or{n}_low", f"_or{n}_open", f"_or{n}_close"])
    return out.drop("_session_date")
