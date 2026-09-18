from __future__ import annotations

import polars as pl


def candlestick_signature_table(
    df: pl.DataFrame,
    *,
    min_occurrences: int = 50,
) -> pl.DataFrame:
    required = {
        "direction",
        "body_pct_range",
        "upper_wick_pct_range",
        "lower_wick_pct_range",
        "label",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    return (
        df.with_columns(
            [
                pl.when(pl.col("body_pct_range") < 0.2)
                .then(0)
                .when(pl.col("body_pct_range") < 0.6)
                .then(1)
                .otherwise(2)
                .alias("_body_bin"),
                pl.when(pl.col("upper_wick_pct_range") < 0.2)
                .then(0)
                .when(pl.col("upper_wick_pct_range") < 0.5)
                .then(1)
                .otherwise(2)
                .alias("_upper_bin"),
                pl.when(pl.col("lower_wick_pct_range") < 0.2)
                .then(0)
                .when(pl.col("lower_wick_pct_range") < 0.5)
                .then(1)
                .otherwise(2)
                .alias("_lower_bin"),
            ]
        )
        .group_by(["direction", "_body_bin", "_upper_bin", "_lower_bin"])
        .agg(
            [
                pl.len().alias("occurrences"),
                (pl.col("label") == 1).mean().alias("long_rate"),
                (pl.col("label") == -1).mean().alias("short_rate"),
                pl.col("label").mean().alias("mean_label"),
            ]
        )
        .filter(pl.col("occurrences") >= min_occurrences)
        .sort("occurrences", descending=True)
    )
