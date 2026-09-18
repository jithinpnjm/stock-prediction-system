from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.backtest.metrics import calculate_metrics


def run() -> None:
    equity = pl.read_parquet("data/backtest/equity_curve.parquet")
    trades_df = pl.read_parquet("data/backtest/trades.parquet")
    from src.backtest.portfolio import Trade

    trades = [
        Trade(
            entry_time=row["entry_time"],
            exit_time=row["exit_time"],
            entry_price=float(row["entry_price"]),
            exit_price=float(row["exit_price"]),
            direction=int(row["direction"]),
            units=float(row["units"]),
            gross_pnl=float(row["gross_pnl"]),
            commissions=float(row["commissions"]),
            transaction_costs=0.0,
            pnl=float(row["pnl"]),
        )
        for row in trades_df.to_dicts()
    ]
    metrics = calculate_metrics(equity, trades)
    Path("artifacts/backtest").mkdir(parents=True, exist_ok=True)
    Path("artifacts/backtest/report.json").write_text(
        json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run()
