from src.backtest.portfolio import Portfolio, Trade


class ExecutionHandler:
    def __init__(
        self, portfolio: Portfolio, slippage: float = 2.0, commission: float = 20.0
    ):
        self.portfolio = portfolio
        self.slippage = slippage  # Points slippage per trade
        self.commission = commission  # Flat fee per trade

    def execute_trade(self, time: str, price: float, signal: int, size: float = 15.0):
        """
        Executes trades based on signals.
        signal: 1 (Buy/Long), -1 (Sell/Short), 0 (Close Position)
        size: Lot size (e.g., BankNifty lot is usually 15)
        """
        # If we already have a position and the signal changes or goes flat, close it
        if self.portfolio.position != 0 and (
            signal != self.portfolio.position or signal == 0
        ):
            self._close_position(time, price)

        # If we have no position and there is an active directional signal, open it
        if self.portfolio.position == 0 and signal in [1, -1]:
            self._open_position(time, price, signal, size)

    def _open_position(self, time: str, price: float, direction: int, size: float):
        execution_price = price + (self.slippage * direction)

        self.portfolio.position = direction
        self.portfolio.position_size = size
        self.portfolio.entry_price = execution_price
        self.portfolio.entry_time = time

        # Deduct commission
        self.portfolio.cash -= self.commission

    def _close_position(self, time: str, price: float):
        direction = self.portfolio.position
        execution_price = price - (self.slippage * direction)

        # Calculate PnL
        price_diff = (execution_price - self.portfolio.entry_price) * direction
        pnl = (price_diff * self.portfolio.position_size) - self.commission

        self.portfolio.cash += (
            self.portfolio.entry_price * self.portfolio.position_size * direction
        )  # free up margin (mocked)
        self.portfolio.cash += pnl

        # Record trade
        trade = Trade(
            entry_time=self.portfolio.entry_time,
            entry_price=self.portfolio.entry_price,
            direction=direction,
            size=self.portfolio.position_size,
            exit_time=time,
            exit_price=execution_price,
            pnl=pnl,
        )
        self.portfolio.trade_history.append(trade)

        # Reset position
        self.portfolio.position = 0
        self.portfolio.position_size = 0.0
        self.portfolio.entry_price = 0.0
        self.portfolio.entry_time = None
