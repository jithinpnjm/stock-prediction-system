import polars as pl
from src.backtest.portfolio import Portfolio
from src.backtest.execution import ExecutionHandler

class EventDrivenBacktester:
    def __init__(self, df: pl.DataFrame, initial_capital: float = 100000.0):
        """
        df must contain: 'datetime', 'close', and 'signal'
        """
        self.df = df
        self.portfolio = Portfolio(initial_capital)
        self.execution = ExecutionHandler(self.portfolio)
        self.equity_curve = []
        
    def run(self):
        # Event Loop
        for row in self.df.iter_rows(named=True):
            dt = str(row['datetime'])
            price = row['close']
            signal = row.get('signal', 0)
            
            # 1. Execute trades based on signal
            self.execution.execute_trade(dt, price, signal)
            
            # 2. Mark to Market
            current_equity = self.portfolio.update_portfolio(price)
            
            # 3. Record Equity Curve
            self.equity_curve.append({
                "datetime": dt,
                "equity": current_equity
            })
            
        # Close any open positions at the end of the backtest
        last_row = self.df.row(-1, named=True)
        if self.portfolio.position != 0:
            self.execution.execute_trade(str(last_row['datetime']), last_row['close'], 0)
            
        return pl.DataFrame(self.equity_curve), self.portfolio.trade_history
