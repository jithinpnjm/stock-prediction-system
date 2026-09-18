from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Trade:
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: int
    units: float
    gross_pnl: float
    commissions: float
    transaction_costs: float
    pnl: float

    @property
    def duration_minutes(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 60.0


class Portfolio:
    def __init__(self, initial_capital: float, *, point_value: float = 1.0):
        self.initial_capital = float(initial_capital)
        self.realized_cash = float(initial_capital)
        self.point_value = float(point_value)
        self.direction = 0
        self.units = 0.0
        self.entry_price = 0.0
        self.entry_time: datetime | None = None
        self.entry_costs = 0.0
        self.trade_history: list[Trade] = []

    def open(self, time: datetime, price: float, direction: int, units: float, costs: float) -> None:
        if self.direction != 0:
            raise RuntimeError("Cannot open while a position is active")
        if direction not in (-1, 1) or units <= 0:
            raise ValueError("direction must be -1/+1 and units must be > 0")
        self.direction = direction
        self.units = float(units)
        self.entry_price = float(price)
        self.entry_time = time
        self.entry_costs = float(costs)
        self.realized_cash -= costs

    def close(self, time: datetime, price: float, costs: float) -> Trade:
        if self.direction == 0 or self.entry_time is None:
            raise RuntimeError("No open position")
        gross = (
            (float(price) - self.entry_price)
            * self.direction
            * self.units
            * self.point_value
        )
        pnl = gross - self.entry_costs - costs
        self.realized_cash += gross - costs
        trade = Trade(
            entry_time=self.entry_time,
            exit_time=time,
            entry_price=self.entry_price,
            exit_price=float(price),
            direction=self.direction,
            units=self.units,
            gross_pnl=gross,
            commissions=self.entry_costs + costs,
            transaction_costs=0.0,
            pnl=pnl,
        )
        self.trade_history.append(trade)
        self.direction = 0
        self.units = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.entry_costs = 0.0
        return trade

    def mark_to_market(self, price: float) -> float:
        unrealized = 0.0
        if self.direction:
            unrealized = (
                (float(price) - self.entry_price)
                * self.direction
                * self.units
                * self.point_value
            )
        return self.realized_cash + unrealized
