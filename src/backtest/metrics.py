import numpy as np
import polars as pl


def calculate_metrics(equity_df: pl.DataFrame, trades: list) -> dict:
    if equity_df.is_empty():
        return {}

    initial_equity = equity_df.select("equity").row(0)[0]
    final_equity = equity_df.select("equity").row(-1)[0]

    total_return = (final_equity - initial_equity) / initial_equity

    # Max Drawdown
    equity_array = equity_df["equity"].to_numpy()
    running_max = np.maximum.accumulate(equity_array)
    drawdowns = (running_max - equity_array) / running_max
    max_drawdown = np.max(drawdowns)

    # Trade Stats
    total_trades = len(trades)
    winning_trades = [t for t in trades if t.pnl > 0]
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

    gross_profit = sum(t.pnl for t in winning_trades)
    gross_loss = sum(abs(t.pnl) for t in trades if t.pnl <= 0)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "Initial Capital": initial_equity,
        "Final Equity": final_equity,
        "Total Return (%)": total_return * 100,
        "Max Drawdown (%)": max_drawdown * 100,
        "Total Trades": total_trades,
        "Win Rate (%)": win_rate * 100,
        "Profit Factor": profit_factor,
    }
