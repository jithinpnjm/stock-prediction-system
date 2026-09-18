from __future__ import annotations

from datetime import timedelta

import polars as pl


def add_label_timing(
    df: pl.DataFrame,
    barrier_timestamp: str | None = None,
) -> pl.DataFrame:
    """Add auditable time-to-event fields.

    When exact barrier timestamps are unavailable from the source path,
    timing remains null rather than being fabricated.
    """
    if barrier_timestamp and barrier_timestamp in df.columns:
        return df.with_columns(
            (
                pl.col(barrier_timestamp) - pl.col("timestamp")
            ).dt.total_seconds().alias("time_to_barrier_seconds")
        )
    return df.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("time_to_barrier_seconds")
    )
