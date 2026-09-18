from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Trade:
    entry_time: object
    exit_time: object
    direction: int
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    exit_reason: str
    signal_time: object


class Portfolio:
    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital=initial_capital
        self.cash=initial_capital
        self.position=0
        self.quantity=0.0
        self.entry_price=0.0
        self.entry_time=None
        self.signal_time=None
        self.trade_history:list[Trade]=[]

    @property
    def equity(self)->float:
        return self.cash

    def open(self,time,signal_time,price,direction,quantity):
        if self.position != 0:
            raise RuntimeError("portfolio already has a position")
        self.position=direction
        self.quantity=quantity
        self.entry_price=price
        self.entry_time=time
        self.signal_time=signal_time

    def close(self,time,price,reason):
        if self.position == 0:
            return None
        pnl=(price-self.entry_price)*self.position*self.quantity
        self.cash+=pnl
        trade=Trade(
            entry_time=self.entry_time,exit_time=time,direction=self.position,
            quantity=self.quantity,entry_price=self.entry_price,exit_price=price,
            pnl=pnl,exit_reason=reason,signal_time=self.signal_time
        )
        self.trade_history.append(trade)
        self.position=0; self.quantity=0.0; self.entry_price=0.0
        self.entry_time=None; self.signal_time=None
        return trade
