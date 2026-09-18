from datetime import datetime,timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.backtest.engine import BacktestConfig,EventDrivenBacktester


def test_signal_uses_future_bar_not_same_timestamp():
    tz=ZoneInfo("Asia/Kolkata")
    times=[datetime(2026,9,15,9,29,tzinfo=tz)+timedelta(minutes=i) for i in range(10)]
    bars=pl.DataFrame({
        "timestamp":times,
        "open":[100.0]*10,"high":[101.0]*10,"low":[99.0]*10,
        "close":[100.0]*10,"volume":[1000.0]*10,
    })
    signal_time=times[1]
    signals=pl.DataFrame({"timestamp":[signal_time],"signal":[1]})
    equity,trades=EventDrivenBacktester(
        BacktestConfig(target_points=200,stop_points=70)
    ).run(signals,bars)
    assert trades
    assert trades[0].signal_time==signal_time
    assert trades[0].entry_time>signal_time
