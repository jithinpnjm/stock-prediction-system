from __future__ import annotations

import polars as pl


def add_consecutive_cluster_features(df: pl.DataFrame) -> pl.DataFrame:
    """Causal run-length of same-direction candles, reset at session start.

    consecutive_length counts the current bar and any immediately
    preceding bars (within the same session) sharing its direction.
    A direction change, or the first bar of a session, resets it to 1.
    """
    out = df.sort("timestamp")
    out = out.with_columns(pl.col("timestamp").dt.date().alias("_session_date"))
    if "f_direction" not in out.columns:
        from .candles import add_candle_geometry_features

        out = add_candle_geometry_features(out)
        out = out.with_columns(pl.col("timestamp").dt.date().alias("_session_date"))

    direction = pl.col("f_direction")
    same_as_prev = direction == direction.shift(1).over("_session_date")
    run_break = (~same_as_prev.fill_null(False)).cast(pl.Int32)
    run_id = run_break.cum_sum().over("_session_date")
    consecutive_length = pl.int_range(pl.len()).over(["_session_date", "run_id"]) + 1

    out = out.with_columns(run_id.alias("run_id")).with_columns(
        consecutive_length.alias("consecutive_length")
    )
    return out.drop(["_session_date", "run_id"])


def add_candle_cluster_features(df: pl.DataFrame, max_bars: int = 10) -> pl.DataFrame:
    out = df.sort("timestamp")
    out = out.with_columns(pl.col("timestamp").dt.date().alias("_session_date"))
    if "f_direction" not in out.columns or "f_return_1" not in out.columns:
        from .candles import add_candle_geometry_features

        out = add_candle_geometry_features(out)
        out = out.with_columns(pl.col("timestamp").dt.date().alias("_session_date"))
    eps = 1e-9
    for n in range(1, max_bars + 1):
        rolling_range = pl.col("f_range").rolling_sum(n).over("_session_date")
        out = out.with_columns(
            pl.col("f_return_1")
            .rolling_sum(n)
            .over("_session_date")
            .alias(f"f_cluster_{n}_return"),
            rolling_range.alias(f"f_cluster_{n}_range"),
            pl.col("f_body").rolling_sum(n).over("_session_date").alias(f"f_cluster_{n}_body"),
            pl.col("f_direction")
            .rolling_sum(n)
            .over("_session_date")
            .alias(f"f_cluster_{n}_direction_balance"),
            (pl.col("f_direction") == 1)
            .cast(pl.Int8)
            .rolling_sum(n)
            .over("_session_date")
            .alias(f"f_cluster_{n}_up_count"),
            (pl.col("f_direction") == -1)
            .cast(pl.Int8)
            .rolling_sum(n)
            .over("_session_date")
            .alias(f"f_cluster_{n}_down_count"),
            pl.col("high").rolling_max(n).over("_session_date").alias(f"f_cluster_{n}_high"),
            pl.col("low").rolling_min(n).over("_session_date").alias(f"f_cluster_{n}_low"),
            (
                (pl.col("close") - pl.col("close").shift(n).over("_session_date"))
                / (rolling_range + eps)
            ).alias(f"f_cluster_{n}_efficiency"),
            pl.col("volume")
            .rolling_mean(n)
            .over("_session_date")
            .alias(f"f_cluster_{n}_volume_mean"),
        )
    return out.drop("_session_date")
