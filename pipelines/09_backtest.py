from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl

from src.backtest.decision import build_trade_decisions
from src.backtest.engine import EventDrivenBacktester
from src.backtest.execution import ExecutionCosts
from src.common.config import load_yaml


def run() -> None:
    cfg = load_yaml(os.getenv("BACKTEST_CONFIG", "configs/backtest/default.yaml"))
    prices = pl.read_parquet("data/silver/5m_canonical.parquet")
    predictions = pl.read_parquet("data/ml/oof_predictions.parquet")
    merged = predictions.join(prices, on="timestamp", how="inner")
    decisions = build_trade_decisions(
        merged,
        min_confidence=float(cfg["min_confidence"]),
        min_edge=float(cfg["min_edge"]),
    )
    bt = EventDrivenBacktester(
        decisions,
        initial_capital=float(cfg["initial_capital"]),
        units=float(cfg["units"]),
        point_value=float(cfg["point_value"]),
        latency_bars=int(cfg["latency_bars"]),
        execution_price_column=cfg["execution_price"],
        costs=ExecutionCosts(
            spread_points=float(cfg["spread_points"]),
            slippage_points=float(cfg["slippage_points"]),
            commission_per_order=float(cfg["commission_per_order"]),
            transaction_cost_bps=float(cfg["transaction_cost_bps"]),
        ),
    )
    equity, trades = bt.run()
    os.makedirs("data/backtest", exist_ok=True)
    equity.write_parquet("data/backtest/equity_curve.parquet", compression="zstd")
    pl.DataFrame(
        [
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "direction": t.direction,
                "units": t.units,
                "gross_pnl": t.gross_pnl,
                "commissions": t.commissions,
                "pnl": t.pnl,
            }
            for t in trades
        ]
    ).write_parquet("data/backtest/trades.parquet", compression="zstd")
    Path("data/backtest/predictions_with_prices.parquet").write_bytes(
        decisions.write_parquet(None) if False else b""
    )
    decisions.write_parquet("data/backtest/predictions_with_prices.parquet", compression="zstd")
    print(f"Backtest produced {len(trades)} trades")


if __name__ == "__main__":
    run()
