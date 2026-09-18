from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.backtest.engine import BacktestConfig, EventDrivenBacktester
from src.backtest.execution import ExecutionConfig


def test_backtest_respects_latency_and_closes_position():
    tz = ZoneInfo("Asia/Kolkata")
    start = datetime(2026, 1, 5, 9, 30, tzinfo=tz)
    ts = [start + timedelta(minutes=i) for i in range(4)]
    bars = pl.DataFrame(
        {
            "timestamp": ts,
            "open": [100, 100, 103, 104],
            "high": [101, 101, 104, 105],
            "low": [99, 99, 102, 103],
            "close": [100, 100, 103, 104],
        }
    )
    signals = pl.DataFrame({"timestamp": [ts[0]], "signal": [1]})
    bt = EventDrivenBacktester(
        BacktestConfig(
            initial_capital=1000,
            target_points=3,
            stop_points=70,
            execution=ExecutionConfig(
                slippage_points=0,
                commission_per_order=0,
                quantity=1,
                latency_bars=1,
            ),
        )
    )
    equity, trades = bt.run(signals, bars)
    assert len(trades) == 1
    assert trades[0].entry_time == ts[1]
    assert trades[0].exit_reason == "target"
    assert equity["equity"][-1] == 1003
