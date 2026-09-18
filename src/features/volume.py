from __future__ import annotations

import polars as pl


def add_volume_features(df: pl.DataFrame) -> pl.DataFrame:
    session_cum_volume = pl.col("volume").cum_sum().over("session_date")
    typical_price = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    session_vwap = (
        (typical_price * pl.col("volume")).cum_sum().over("session_date")
        / session_cum_volume.clip(lower_bound=1.0)
    )
    return df.with_columns(
        [
            (
                pl.col("volume")
                / pl.col("volume").shift(1).rolling_mean(20)
            ).alias("volume_vs_prior_20"),
            session_vwap.alias("session_vwap"),
            (
                (pl.col("close") - session_vwap)
                / (pl.col("atr_14") + 1e-9)
            ).alias("distance_vwap_atr"),
            (
                pl.col("volume").cum_sum().over("session_date")
            ).alias("session_cumulative_volume"),
        ]
    )
