from __future__ import annotations

import math
import numpy as np
import polars as pl

from .portfolio import Trade


def calculate_metrics(equity_df:pl.DataFrame,trades:list[Trade],periods_per_year:int=252)->dict:
    if equity_df.is_empty():return {}
    daily=(
        equity_df.with_columns(pl.col("timestamp").dt.date().alias("_date"))
        .group_by("_date").agg(pl.col("equity").last().alias("equity"))
        .sort("_date")["equity"].to_numpy()
    )
    initial=float(daily[0]); final=float(daily[-1])
    running=np.maximum.accumulate(daily)
    dd=(running-daily)/np.maximum(running,1e-12)
    returns=np.diff(daily)/np.maximum(daily[:-1],1e-12) if len(daily)>1 else np.array([])
    mean=float(returns.mean()) if len(returns) else 0.0
    std=float(returns.std(ddof=1)) if len(returns)>1 else 0.0
    sharpe=mean/std*math.sqrt(periods_per_year) if std>0 else 0.0
    downside=returns[returns<0]
    dstd=float(downside.std(ddof=1)) if len(downside)>1 else 0.0
    sortino=mean/dstd*math.sqrt(periods_per_year) if dstd>0 else 0.0
    years=max((len(daily)-1)/periods_per_year,1/periods_per_year)
    cagr=(final/initial)**(1/years)-1 if initial>0 else 0.0
    max_dd=float(dd.max()) if len(dd) else 0.0
    wins=[t.pnl for t in trades if t.pnl>0]
    losses=[t.pnl for t in trades if t.pnl<=0]
    gross_profit=sum(wins); gross_loss=abs(sum(losses))
    return {
        "initial_capital":initial,"final_equity":final,
        "total_return_pct":(final/initial-1)*100 if initial else 0.0,
        "cagr_pct":cagr*100,"max_drawdown_pct":max_dd*100,
        "total_trades":len(trades),
        "win_rate_pct":len(wins)/len(trades)*100 if trades else 0.0,
        "profit_factor":gross_profit/gross_loss if gross_loss else float("inf"),
        "expectancy_per_trade":float(np.mean([t.pnl for t in trades])) if trades else 0.0,
        "sharpe":sharpe,"sortino":sortino,
        "calmar":cagr/max_dd if max_dd else float("inf"),
        "avg_win":gross_profit/len(wins) if wins else 0.0,
        "avg_loss":-gross_loss/len(losses) if losses else 0.0,
    }
