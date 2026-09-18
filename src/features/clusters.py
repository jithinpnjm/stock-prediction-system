from __future__ import annotations

import polars as pl


def add_candle_cluster_features(
    df: pl.DataFrame,
    max_bars: int = 10,
) -> pl.DataFrame:
    """Add causal rolling descriptors for consecutive 5m candle clusters."""
    out = df.sort("timestamp")
    if "f_direction" not in out.columns:
        from .candles import add_candle_geometry_features
        out = add_candle_geometry_features(out)
    if "f_return_1" not in out.columns:
        out = out.with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("f_return_1")
        )
    eps=1e-9
    for n in range(1, max_bars + 1):
        out = out.with_columns(
            pl.col("f_return_1").rolling_sum(n).alias(f"f_cluster_{n}_return"),
            pl.col("f_range").rolling_sum(n).alias(f"f_cluster_{n}_range"),
            pl.col("f_body").rolling_sum(n).alias(f"f_cluster_{n}_body"),
            pl.col("f_direction").rolling_sum(n).alias(f"f_cluster_{n}_direction_balance"),
            (pl.col("f_direction") == 1).cast(pl.Int8).rolling_sum(n).alias(
                f"f_cluster_{n}_up_count"
            ),
            (pl.col("f_direction") == -1).cast(pl.Int8).rolling_sum(n).alias(
                f"f_cluster_{n}_down_count"
            ),
            pl.col("high").rolling_max(n).alias(f"f_cluster_{n}_high"),
            pl.col("low").rolling_min(n).alias(f"f_cluster_{n}_low"),
            (
                (pl.col("close") - pl.col("close").shift(n))
                / (pl.col("f_range").rolling_sum(n) + eps)
            ).alias(f"f_cluster_{n}_efficiency"),
            (
                pl.col("volume").rolling_mean(n)
            ).alias(f"f_cluster_{n}_volume_mean"),
        )
    return out
