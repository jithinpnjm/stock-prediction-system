from __future__ import annotations

import math
import numpy as np
import polars as pl


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peak, 1e-12) - 1.0
    return float(abs(dd.min()))


def calculate_metrics(
    equity_df: pl.DataFrame,
    trades: list,
    *,
    bars_per_day: int = 75,
    trading_days_per_year: int = 252,
) -> dict[str, float]:
    if equity_df.is_empty():
        return {}
    equity = equity_df["equity"].cast(pl.Float64).to_numpy()
    initial = float(equity[0])
    final = float(equity[-1])
    total_return = final / initial - 1.0 if initial else 0.0
    years = max(len(equity) / (bars_per_day * trading_days_per_year), 1 / trading_days_per_year)
    cagr = (final / initial) ** (1 / years) - 1.0 if initial > 0 and final > 0 else -1.0

    ret = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    ann = (bars_per_day * trading_days_per_year) ** 0.5
    sharpe = float(np.mean(ret) / np.std(ret, ddof=1) * ann) if len(ret) > 1 and np.std(ret) > 0 else 0.0
    downside = ret[ret < 0]
    sortino = float(np.mean(ret) / np.std(downside, ddof=1) * ann) if len(downside) > 1 and np.std(downside) > 0 else 0.0

    pnl = np.array([float(t.pnl) for t in trades], dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0

    return {
        "initial_equity": initial,
        "final_equity": final,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": _max_drawdown(equity),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": cagr / _max_drawdown(equity) if _max_drawdown(equity) > 0 else 0.0,
        "total_trades": float(len(trades)),
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else 0.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else math.inf,
        "expectancy_per_trade": float(pnl.mean()) if len(pnl) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "median_duration_minutes": float(np.median([t.duration_minutes for t in trades])) if trades else 0.0,
    }
