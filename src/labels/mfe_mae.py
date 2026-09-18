from __future__ import annotations

import polars as pl


def add_excursion_features(df: pl.DataFrame) -> pl.DataFrame:
    """Derive selected-direction excursion diagnostics from label columns."""
    if "label" not in df.columns:
        raise ValueError("label column is required")
    return df.with_columns(
        pl.when(pl.col("label") == 1)
        .then(pl.col("mfe_long_points"))
        .when(pl.col("label") == -1)
        .then(pl.col("mfe_short_points"))
        .otherwise(pl.max_horizontal("mfe_long_points", "mfe_short_points"))
        .alias("mfe_points"),
        pl.when(pl.col("label") == 1)
        .then(pl.col("mae_long_points"))
        .when(pl.col("label") == -1)
        .then(pl.col("mae_short_points"))
        .otherwise(pl.max_horizontal("mae_long_points", "mae_short_points"))
        .alias("mae_points"),
    )
