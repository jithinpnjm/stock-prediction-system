from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionConfig:
    slippage_points: float = 2.0
    commission_per_order: float = 20.0
    quantity: float = 15.0
    latency_bars: int = 1


def apply_entry_slippage(price: float,direction: int,points: float)->float:
    return price + direction*points


def apply_exit_slippage(price: float,direction: int,points: float)->float:
    return price - direction*points


def charge_order_fee(cash: float,fee: float)->float:
    return cash-fee
