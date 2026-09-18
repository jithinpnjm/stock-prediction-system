from __future__ import annotations

import polars as pl


def add_volatility_features(
    df: pl.DataFrame,
    periods: tuple[int, ...] = (6, 14, 30),
) -> pl.DataFrame:
    out = df.sort("timestamp").with_columns(
        pl.col("close").shift(1).alias("_prev_close")
    ).with_columns(
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("_prev_close")).abs(),
            (pl.col("low") - pl.col("_prev_close")).abs(),
        ).alias("_tr")
    )
    eps=1e-9
    for p in periods:
        out = out.with_columns(
            pl.col("_tr").rolling_mean(p).alias(f"f_atr_{p}"),
            pl.col("close").pct_change(p).abs().alias(f"f_abs_return_{p}"),
            pl.col("_tr").rolling_std(p).alias(f"f_tr_std_{p}"),
            (pl.col("_tr") / (pl.col("_tr").rolling_mean(p) + eps)).alias(
                f"f_range_to_atr_{p}"
            ),
        ).with_columns(
            (pl.col(f"f_atr_{p}") / (pl.col("close") + eps) * 10_000).alias(
                f"f_natr_{p}_bps"
            )
        )
    out = out.with_columns(
        (pl.col("_tr") / (pl.col("_tr").rolling_mean(14) + eps)).alias(
            "f_volatility_shock"
        )
    )
    return out.drop(["_prev_close", "_tr"])
