from __future__ import annotations

import numpy as np
import polars as pl


def regime_performance(
    predictions: pl.DataFrame,
    *,
    regime_column: str = "regime",
    pnl_column: str = "pnl",
) -> pl.DataFrame:
    if regime_column not in predictions.columns or pnl_column not in predictions.columns:
        raise ValueError("Required regime/pnl columns are missing")
    return (
        predictions.group_by(regime_column)
        .agg(
            [
                pl.len().alias("observations"),
                pl.col(pnl_column).mean().alias("mean_pnl"),
                (pl.col(pnl_column) > 0).mean().alias("win_rate"),
                pl.col(pnl_column).sum().alias("total_pnl"),
            ]
        )
        .sort(regime_column)
    )


def cost_stress_returns(
    returns: np.ndarray,
    turnover: np.ndarray,
    cost_bps: float,
) -> np.ndarray:
    return np.asarray(returns) - (np.asarray(turnover) * cost_bps / 10_000.0)


def parameter_sensitivity(results: pl.DataFrame, parameter: str, metric: str) -> pl.DataFrame:
    return results.group_by(parameter).agg(
        [
            pl.col(metric).mean().alias(f"{metric}_mean"),
            pl.col(metric).std().alias(f"{metric}_std"),
            pl.len().alias("runs"),
        ]
    ).sort(parameter)
