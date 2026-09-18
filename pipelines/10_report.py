from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.backtest.metrics import calculate_metrics


def run():
    validation=json.loads(Path("reports/validation_report.json").read_text())
    equity=pl.read_parquet("data/backtest/equity_curve.parquet")
    trades_df=pl.read_parquet("data/backtest/trades.parquet")
    from src.backtest.portfolio import Trade
    trades=[
        Trade(
            entry_time=r["entry_time"],exit_time=r["exit_time"],direction=int(r["direction"]),
            quantity=float(r["quantity"]),entry_price=float(r["entry_price"]),
            exit_price=float(r["exit_price"]),pnl=float(r["pnl"]),
            exit_reason=r["exit_reason"],signal_time=r["signal_time"]
        )
        for r in trades_df.to_dicts()
    ]
    backtest=calculate_metrics(equity,trades)
    report={"validation":validation,"backtest":backtest}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/research_report.json").write_text(json.dumps(report,indent=2,default=str)+"\n")
    print(json.dumps(report,indent=2,default=str))


if __name__=="__main__":
    run()
