from __future__ import annotations

import polars as pl


def add_barrier_timing_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            (
                (pl.col("barrier_time") - pl.col("entry_time")).dt.total_seconds() / 60.0
            ).alias("time_to_barrier_min"),
            pl.col("path_complete").cast(pl.Int8).alias("label_path_complete"),
        ]
    )
