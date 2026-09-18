from __future__ import annotations

import polars as pl

from src.models.ensemble import decision_from_probabilities, expected_value_per_point


def build_trade_decisions(
    predictions: pl.DataFrame,
    *,
    min_confidence: float = 0.55,
    min_edge: float = 0.08,
    target_points: float = 200.0,
    stop_points: float = 70.0,
) -> pl.DataFrame:
    required = {"timestamp", "p_short", "p_flat", "p_long"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    probs = predictions.select(["p_short", "p_flat", "p_long"]).to_numpy()
    ev_long, ev_short = expected_value_per_point(
        probs,
        target_points=target_points,
        stop_points=stop_points,
    )
    signal = decision_from_probabilities(
        probs,
        min_confidence=min_confidence,
        min_edge=min_edge,
    )

    return predictions.with_columns(
        [
            pl.Series("ev_long_points", ev_long),
            pl.Series("ev_short_points", ev_short),
            pl.Series("raw_signal", signal),
        ]
    ).with_columns(
        pl.when((pl.col("raw_signal") == 1) & (pl.col("ev_long_points") > 0))
        .then(1)
        .when((pl.col("raw_signal") == -1) & (pl.col("ev_short_points") > 0))
        .then(-1)
        .otherwise(0)
        .alias("signal"),
    )
