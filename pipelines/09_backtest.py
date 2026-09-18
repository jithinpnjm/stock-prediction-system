from __future__ import annotations

import os

import polars as pl

from src.backtest.decision import build_trade_decisions
from src.backtest.engine import EventDrivenBacktester
from src.backtest.execution import ExecutionCosts
from src.common.config import load_yaml


def run() -> None:
    cfg = load_yaml(
        os.getenv(
            "BACKTEST_CONFIG",
            "configs/backtest/default.yaml",
        )
    )

    prices = pl.read_parquet(
        "data/silver/5m_canonical.parquet"
    )
    predictions = pl.read_parquet(
        "data/ml/calibrated_oof_predictions.parquet"
    )

    merged = (
        predictions
        .join(prices, on="timestamp", how="inner")
        .sort("timestamp")
    )

    if merged.is_empty():
        raise RuntimeError(
            "No timestamp overlap between predictions and prices"
        )

    decisions = build_trade_decisions(
        merged,
        min_confidence=float(cfg["min_confidence"]),
        min_edge=float(cfg["min_edge"]),
    )

    backtester = EventDrivenBacktester(
        decisions,
        initial_capital=float(cfg["initial_capital"]),
        units=float(cfg["units"]),
        point_value=float(cfg["point_value"]),
        latency_bars=int(cfg["latency_bars"]),
        execution_price_column=cfg["execution_price"],
        costs=ExecutionCosts(
            spread_points=float(
                cfg["spread_points"]
            ),
            slippage_points=float(
                cfg["slippage_points"]
            ),
            commission_per_order=float(
                cfg["commission_per_order"]
            ),
            transaction_cost_bps=float(
                cfg["transaction_cost_bps"]
            ),
        ),
        allow_overnight=bool(
            cfg.get("allow_overnight", False)
        ),
    )

    equity, trades = backtester.run()

    os.makedirs("data/backtest", exist_ok=True)
    equity.write_parquet(
        "data/backtest/equity_curve.parquet",
        compression="zstd",
    )

    pl.DataFrame(
        [
            {
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "direction": trade.direction,
                "units": trade.units,
                "gross_pnl": trade.gross_pnl,
                "commissions": trade.commissions,
                "transaction_costs": trade.transaction_costs,
                "pnl": trade.pnl,
                "exit_reason": trade.exit_reason,
            }
            for trade in trades
        ]
    ).write_parquet(
        "data/backtest/trades.parquet",
        compression="zstd",
    )

    decisions.write_parquet(
        "data/backtest/predictions_with_prices.parquet",
        compression="zstd",
    )

    print(
        f"Backtest produced {len(trades)} trades"
    )


if __name__ == "__main__":
    run()
