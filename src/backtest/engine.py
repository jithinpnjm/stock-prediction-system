from __future__ import annotations

from collections import deque

import polars as pl

from src.backtest.execution import ExecutionCosts, ExecutionHandler
from src.backtest.portfolio import Portfolio


class EventDrivenBacktester:
    def __init__(
        self,
        df: pl.DataFrame,
        *,
        initial_capital: float = 100_000.0,
        units: float = 1.0,
        point_value: float = 1.0,
        latency_bars: int = 1,
        execution_price_column: str = "open",
        costs: ExecutionCosts | None = None,
    ):
        required = {"timestamp", "open", "high", "low", "close", "signal"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing backtest columns: {sorted(missing)}")
        if latency_bars < 0:
            raise ValueError("latency_bars cannot be negative")
        if execution_price_column not in {"open", "close"}:
            raise ValueError("execution_price_column must be open or close")
        self.df = df.sort("timestamp")
        self.units = float(units)
        self.latency_bars = latency_bars
        self.execution_price_column = execution_price_column
        self.portfolio = Portfolio(initial_capital, point_value=point_value)
        self.execution = ExecutionHandler(
            self.portfolio, costs or ExecutionCosts()
        )

    def run(self) -> tuple[pl.DataFrame, list]:
        rows = self.df.to_dicts()
        pending: deque[tuple[int, int]] = deque()
        equity = []

        for i, row in enumerate(rows):
            signal = int(row["signal"])
            execute_at = i + self.latency_bars
            if execute_at < len(rows):
                pending.append((execute_at, signal))

            while pending and pending[0][0] == i:
                _, target = pending.popleft()
                market_price = float(row[self.execution_price_column])
                self.execution.execute(
                    time=row["timestamp"],
                    market_price=market_price,
                    target_direction=target,
                    units=self.units,
                )

            equity.append(
                {
                    "timestamp": row["timestamp"],
                    "equity": self.portfolio.mark_to_market(float(row["close"])),
                    "position": self.portfolio.direction,
                    "units": self.portfolio.units,
                }
            )

        if rows and self.portfolio.direction:
            row = rows[-1]
            market_price = float(row["close"])
            self.execution.execute(
                time=row["timestamp"],
                market_price=market_price,
                target_direction=0,
                units=self.units,
            )
            equity[-1]["equity"] = self.portfolio.mark_to_market(market_price)

        return pl.DataFrame(equity), self.portfolio.trade_history
