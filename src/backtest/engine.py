from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
import polars as pl

from .execution import ExecutionConfig,apply_entry_slippage,apply_exit_slippage
from .portfolio import Portfolio


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital:float=100_000.0
    target_points:float=200.0
    stop_points:float=70.0
    entry_start:str="09:30"
    entry_end:str="15:00"
    flatten_time:str="15:30"
    execution:ExecutionConfig=ExecutionConfig()


class EventDrivenBacktester:
    """Execute completed-5m signals on subsequent 1m bars with explicit cost assumptions."""

    def __init__(self,config:BacktestConfig|None=None):
        self.config=config or BacktestConfig()

    def run(self,signals:pl.DataFrame,bars_1m:pl.DataFrame):
        if not {"timestamp","signal"}.issubset(signals.columns):
            raise ValueError("signals must contain timestamp and signal")
        bars=bars_1m.sort("timestamp").to_dicts()
        sigs=signals.sort("timestamp").to_dicts()
        if not bars:return pl.DataFrame(),[]

        timestamps=[b["timestamp"] for b in bars]
        scheduled={}
        for s in sigs:
            st=s["timestamp"]; signal=int(s["signal"])
            idx=bisect_right(timestamps,st)+self.config.execution.latency_bars-1
            if 0<=idx<len(bars):
                scheduled[idx]=(st,signal)

        portfolio=Portfolio(self.config.initial_capital)
        curve=[]
        for idx,row in enumerate(bars):
            ts=row["timestamp"]; clock=ts.strftime("%H:%M")
            if idx in scheduled:
                signal_time,signal=scheduled[idx]
                if signal not in (-1,0,1):
                    raise ValueError("signal must be -1, 0 or 1")
                if not (self.config.entry_start<=clock<=self.config.entry_end):
                    signal=0

                if portfolio.position and (signal==0 or signal!=portfolio.position):
                    px=apply_exit_slippage(
                        float(row["open"]),portfolio.position,
                        self.config.execution.slippage_points
                    )
                    portfolio.close(ts,px,"signal_change")
                    portfolio.cash-=self.config.execution.commission_per_order

                if portfolio.position==0 and signal in (-1,1):
                    px=apply_entry_slippage(
                        float(row["open"]),signal,
                        self.config.execution.slippage_points
                    )
                    execution_time=row.get("source_timestamp",ts)
                    portfolio.open(
                        execution_time,signal_time,px,signal,
                        self.config.execution.quantity
                    )
                    portfolio.cash-=self.config.execution.commission_per_order

            if portfolio.position:
                direction=portfolio.position
                target=portfolio.entry_price+self.config.target_points*direction
                stop=portfolio.entry_price-self.config.stop_points*direction
                high=float(row["high"]); low=float(row["low"])
                hit_target=(high>=target if direction==1 else low<=target)
                hit_stop=(low<=stop if direction==1 else high>=stop)
                if hit_target and hit_stop:
                    px=apply_exit_slippage(stop,direction,self.config.execution.slippage_points)
                    portfolio.close(ts,px,"stop_ambiguous")
                    portfolio.cash-=self.config.execution.commission_per_order
                elif hit_stop:
                    px=apply_exit_slippage(stop,direction,self.config.execution.slippage_points)
                    portfolio.close(ts,px,"stop")
                    portfolio.cash-=self.config.execution.commission_per_order
                elif hit_target:
                    px=apply_exit_slippage(target,direction,self.config.execution.slippage_points)
                    portfolio.close(ts,px,"target")
                    portfolio.cash-=self.config.execution.commission_per_order

            equity=portfolio.cash
            if portfolio.position:
                equity+=(float(row["close"])-portfolio.entry_price)*portfolio.position*portfolio.quantity
            curve.append({"timestamp":ts,"equity":equity})

            if clock>=self.config.flatten_time and portfolio.position:
                px=apply_exit_slippage(
                    float(row["close"]),portfolio.position,
                    self.config.execution.slippage_points
                )
                portfolio.close(ts,px,"session_close")
                portfolio.cash-=self.config.execution.commission_per_order

        return pl.DataFrame(curve),portfolio.trade_history
