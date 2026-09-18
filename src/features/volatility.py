from __future__ import annotations

import polars as pl


def add_volatility_features(
    df: pl.DataFrame,
    *,
    atr_periods: tuple[int, ...] = (5, 14, 30),
    return_periods: tuple[int, ...] = (1, 3, 6, 12, 24),
) -> pl.DataFrame:
    out = df.with_columns(
        [
            (pl.col("high") - pl.col("low")).alias("_tr0"),
            (pl.col("high") - pl.col("close").shift(1)).abs().alias("_tr1"),
            (pl.col("low") - pl.col("close").shift(1)).abs().alias("_tr2"),
        ]
    ).with_columns(pl.max_horizontal("_tr0", "_tr1", "_tr2").alias("true_range"))

    exprs: list[pl.Expr] = []
    for p in atr_periods:
        atr = pl.col("true_range").rolling_mean(p).alias(f"atr_{p}")
        exprs.extend(
            [
                atr,
                (pl.col("true_range") / (pl.col("true_range").rolling_mean(p) + 1e-9)).alias(
                    f"range_to_atr_{p}"
                ),
                (
                    pl.col("true_range").rolling_mean(p)
                    / (pl.col("close").abs() + 1e-9)
                    * 10_000
                ).alias(f"atr_bps_{p}"),
            ]
        )
    for p in return_periods:
        exprs.extend(
            [
                (pl.col("close") / pl.col("close").shift(p) - 1.0).alias(f"return_{p}"),
                (
                    pl.col("close").log().diff().rolling_std(p) * (p**0.5)
                ).alias(f"realized_vol_{p}"),
                (
                    pl.col("high").rolling_max(p) - pl.col("low").rolling_min(p)
                ).alias(f"rolling_range_{p}"),
            ]
        )
    out = out.with_columns(exprs)
    return out.drop(["_tr0", "_tr1", "_tr2", "true_range"])
