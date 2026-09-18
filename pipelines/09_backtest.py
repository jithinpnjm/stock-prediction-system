from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import polars as pl
import yaml

from src.backtest.engine import BacktestConfig, EventDrivenBacktester
from src.backtest.execution import ExecutionConfig
from src.inference.decision import decide


def run():
    cfg = yaml.safe_load(Path("configs/backtest/default.yaml").read_text())
    pred = pl.read_parquet("data/predictions/lgbm_oof.parquet").sort("timestamp")

    decisions = [
        decide(
            float(row["p_short"]),
            float(row["p_none"]),
            float(row["p_long"]),
            target_points=float(cfg["target_points"]),
            stop_points=float(cfg["stop_points"]),
            min_probability=float(cfg["min_probability"]),
            min_edge=float(cfg["min_directional_edge"]),
        )
        for row in pred.to_dicts()
    ]

    signals_df = pred.select("timestamp").with_columns(
        pl.Series("signal", [item.signal for item in decisions], dtype=pl.Int8),
        pl.Series(
            "confidence",
            [item.probability for item in decisions],
            dtype=pl.Float64,
        ),
        pl.Series(
            "expected_value_points",
            [item.expected_value_points for item in decisions],
            dtype=pl.Float64,
        ),
        pl.Series(
            "decision_reason",
            [item.reason for item in decisions],
            dtype=pl.String,
        ),
    )

    bars = pl.read_parquet("data/bronze/validated_1m.parquet")
    bt_cfg = BacktestConfig(
        initial_capital=float(cfg["initial_capital"]),
        target_points=float(cfg["target_points"]),
        stop_points=float(cfg["stop_points"]),
        entry_start=cfg["entry_start"],
        entry_end=cfg["entry_end"],
        flatten_time=cfg["flatten_time"],
        execution=ExecutionConfig(
            slippage_points=float(cfg["slippage_points"]),
            commission_per_order=float(cfg["commission_per_order"]),
            quantity=float(cfg["quantity"]),
            latency_bars=int(cfg["latency_bars"]),
        ),
    )

    equity, trades = EventDrivenBacktester(bt_cfg).run(signals_df, bars)
    Path("data/backtest").mkdir(parents=True, exist_ok=True)
    equity.write_parquet("data/backtest/equity_curve.parquet")

    if trades:
        pl.DataFrame([asdict(trade) for trade in trades]).write_parquet(
            "data/backtest/trades.parquet"
        )
    else:
        pl.DataFrame(
            {
                "entry_time": pl.Series([], dtype=pl.Datetime("ns", time_zone="Asia/Kolkata")),
                "exit_time": pl.Series([], dtype=pl.Datetime("ns", time_zone="Asia/Kolkata")),
                "direction": pl.Series([], dtype=pl.Int8),
                "quantity": pl.Series([], dtype=pl.Float64),
                "entry_price": pl.Series([], dtype=pl.Float64),
                "exit_price": pl.Series([], dtype=pl.Float64),
                "pnl": pl.Series([], dtype=pl.Float64),
                "exit_reason": pl.Series([], dtype=pl.String),
                "signal_time": pl.Series([], dtype=pl.Datetime("ns", time_zone="Asia/Kolkata")),
            }
        ).write_parquet("data/backtest/trades.parquet")

    print(f"backtest trades={len(trades)}")


if __name__ == "__main__":
    run()
