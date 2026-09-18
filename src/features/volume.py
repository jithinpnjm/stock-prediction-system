from __future__ import annotations

import polars as pl


def add_volume_features(df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """Legacy volume features kept session-local for safety.

    The canonical feature pipeline currently uses dedicated session VWAP and
    cluster features. This module remains available for older experiments.
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    out = df.sort("timestamp").with_columns(
        pl.col("timestamp").dt.date().alias("_session_date"),
    )
    session_cum_volume = pl.col("volume").cum_sum().over("_session_date")
    typical_price = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    session_vwap = (
        (typical_price * pl.col("volume")).cum_sum().over("_session_date")
        / session_cum_volume.clip(lower_bound=1.0)
    )
    prior_volume_mean = (
        pl.col("volume")
        .shift(1)
        .over("_session_date")
        .rolling_mean(lookback)
        .over("_session_date")
    )

    return out.with_columns(
        [
            (pl.col("volume") / (prior_volume_mean + 1e-9)).alias(
                "volume_vs_prior_20"
            ),
            session_vwap.alias("session_vwap"),
            (
                (pl.col("close") - session_vwap)
                / (pl.col("atr_14") + 1e-9)
            ).alias("distance_vwap_atr"),
            session_cum_volume.alias("session_cumulative_volume"),
        ]
    ).drop("_session_date")
