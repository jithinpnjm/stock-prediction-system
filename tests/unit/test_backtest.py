from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.backtest.engine import BacktestConfig, EventDrivenBacktester
from src.backtest.execution import ExecutionConfig


def test_backtest_respects_signal_to_next_1m_source_execution():
    tz = ZoneInfo("Asia/Kolkata")
    signal_time = datetime(2026, 1, 5, 9, 30, tzinfo=tz)
    availability = [
        signal_time + timedelta(minutes=1),
        signal_time + timedelta(minutes=2),
    ]
    source = [
        signal_time,
        signal_time + timedelta(minutes=1),
    ]
    bars = pl.DataFrame(
        {
            "timestamp": availability,
            "source_timestamp": source,
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
        }
    )
    signals = pl.DataFrame(
        {
            "timestamp": [signal_time],
            "signal": [1],
        }
    )
    config = BacktestConfig(
        initial_capital=1000.0,
        target_points=2.0,
        stop_points=70.0,
        entry_start="09:30",
        entry_end="15:00",
        flatten_time="15:30",
        execution=ExecutionConfig(
            slippage_points=0.0,
            commission_per_order=0.0,
            quantity=1.0,
            latency_bars=1,
        ),
    )

    equity, trades = EventDrivenBacktester(config).run(signals, bars)

    assert len(trades) == 1
    assert trades[0].entry_time == source[1]
    assert trades[0].signal_time == signal_time
    assert trades[0].exit_time == availability[1]
    assert trades[0].exit_reason == "target"
    assert equity["equity"][-1] == 1002.0
