from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Trade:
    entry_time: str
    entry_price: float
    direction: int  # 1 for long, -1 for short
    size: float
    exit_time: str = None
    exit_price: float = None
    pnl: float = 0.0

class Portfolio:
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0  # +1 (Long), -1 (Short), 0 (Flat)
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.trade_history: List[Trade] = []

    def update_portfolio(self, current_price: float) -> float:
        """Returns the mark-to-market value of the portfolio."""
        if self.position == 0:
            return self.cash
        
        # Calculate unrealized PnL
        unrealized_pnl = (current_price - self.entry_price) * self.position * self.position_size
        return self.cash + unrealized_pnl
