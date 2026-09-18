from __future__ import annotations

import math

import numpy as np
import polars as pl


def _max_consecutive(
    values: list[bool],
    target: bool,
) -> int:
    best = 0
    current = 0
    for value in values:
        if value == target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def calculate_metrics(
    equity_df: pl.DataFrame,
    trades: list,
) -> dict[str, float | int | None]:
    if equity_df.is_empty():
        return {}

    equity = equity_df.sort("timestamp")

    initial = float(equity["equity"][0])
    final = float(equity["equity"][-1])
    total_return = (
        final / initial - 1.0
        if initial
        else math.nan
    )

    running_max = (
        equity["equity"]
        .cum_max()
    )
    drawdown = (
        1.0
        - equity["equity"]
        / running_max
    )
    max_drawdown = float(
        drawdown.max()
    )

    daily = (
        equity
        .group_by("session_date")
        .agg(
            pl.col("equity")
            .last()
            .alias("daily_equity")
        )
        .sort("session_date")
    )
    daily_returns = (
        daily["daily_equity"]
        .pct_change()
        .drop_nulls()
        .to_numpy()
    )

    sharpe = None
    sortino = None
    if len(daily_returns) >= 2:
        daily_mean = float(
            np.mean(daily_returns)
        )
        daily_std = float(
            np.std(
                daily_returns,
                ddof=1,
            )
        )
        downside = daily_returns[
            daily_returns < 0
        ]
        downside_std = (
            float(
                np.std(
                    downside,
                    ddof=1,
                )
            )
            if len(downside) > 1
            else 0.0
        )

        if daily_std > 0:
            sharpe = (
                daily_mean
                / daily_std
                * math.sqrt(252)
            )
        if downside_std > 0:
            sortino = (
                daily_mean
                / downside_std
                * math.sqrt(252)
            )

    days = max(
        (
            equity["timestamp"][-1]
            - equity["timestamp"][0]
        ).total_seconds()
        / 86_400.0,
        1.0,
    )
    years = days / 365.25
    cagr = (
        (final / initial) ** (1.0 / years)
        - 1.0
        if initial > 0
        and final > 0
        else None
    )

    calmar = (
        cagr / max_drawdown
        if cagr is not None
        and max_drawdown > 0
        else None
    )

    pnls = [float(t.pnl) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else None
    )

    win_rate = (
        len(wins) / len(pnls)
        if pnls
        else None
    )

    average_trade = (
        float(np.mean(pnls))
        if pnls
        else None
    )
    average_win = (
        float(np.mean(wins))
        if wins
        else None
    )
    average_loss = (
        float(np.mean(losses))
        if losses
        else None
    )

    return {
        "initial_capital": initial,
        "final_equity": final,
        "total_return_pct": (
            total_return * 100.0
        ),
        "cagr_pct": (
            cagr * 100.0
            if cagr is not None
            else None
        ),
        "max_drawdown_pct": (
            max_drawdown * 100.0
        ),
        "sharpe_daily": sharpe,
        "sortino_daily": sortino,
        "calmar": calmar,
        "total_trades": len(trades),
        "win_rate_pct": (
            win_rate * 100.0
            if win_rate is not None
            else None
        ),
        "profit_factor": profit_factor,
        "average_trade": average_trade,
        "average_win": average_win,
        "average_loss": average_loss,
        "max_consecutive_wins": (
            _max_consecutive(
                [p > 0 for p in pnls],
                True,
            )
            if pnls
            else 0
        ),
        "max_consecutive_losses": (
            _max_consecutive(
                [p <= 0 for p in pnls],
                True,
            )
            if pnls
            else 0
        ),
    }
