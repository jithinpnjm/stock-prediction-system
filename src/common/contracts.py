from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from enum import StrEnum


MARKET_TIMEZONE = "Asia/Kolkata"
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)
EXPECTED_1M_BARS = 375
EXPECTED_5M_BARS = 75


class BarrierType(StrEnum):
    LONG_TARGET = "long_target"
    LONG_STOP = "long_stop"
    SHORT_TARGET = "short_target"
    SHORT_STOP = "short_stop"
    TIME = "time"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class LabelConfig:
    target_points: float = 200.0
    stop_points: float = 70.0
    horizon_bars: int = 75
    entry_delay_minutes: int = 0


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    lot_size: float = 1.0
    point_value: float = 1.0
    spread_points: float = 0.0
    slippage_points: float = 2.0
    latency_bars: int = 1
    commission_per_order: float = 20.0
    transaction_cost_bps: float = 0.0
