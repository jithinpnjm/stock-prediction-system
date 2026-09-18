from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.backtest.engine import EventDrivenBacktester
from src.backtest.execution import ExecutionCosts


def test_backtest_respects_latency_and_closes_position():
    tz = ZoneInfo("Asia/Kolkata")
    start = datetime(2026, 1, 5, 9, 15, tzinfo=tz)
    ts = [start + timedelta(minutes=5 * i) for i in range(4)]
    df = pl.DataFrame(
        {
            "timestamp": ts,
            "open": [100, 101, 103, 104],
            "high": [101, 102, 104, 105],
            "low": [99, 100, 102, 103],
            "close": [100, 101, 103, 104],
            "signal": [1, 0, 0, 0],
        }
    )
    bt = EventDrivenBacktester(
        df,
        initial_capital=1000,
        units=1,
        point_value=1,
        latency_bars=1,
        execution_price_column="open",
        costs=ExecutionCosts(
            spread_points=0,
            slippage_points=0,
            commission_per_order=0,
            transaction_cost_bps=0,
        ),
    )
    equity, trades = bt.run()
    assert len(trades) == 1
    assert trades[0].entry_time == ts[1]
    assert equity["equity"][-1] == 1003
