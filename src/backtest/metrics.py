from __future__ import annotations

import math
import numpy as np
import polars as pl

from .portfolio import Trade


def calculate_metrics(equity_df:pl.DataFrame,trades:list[Trade],periods_per_year:int=252)->dict:
    if equity_df.is_empty(): return {}
    e=equity_df["equity"].cast(pl.Float64).to_numpy()
    initial=float(e[0]); final=float(e[-1])
    running=np.maximum.accumulate(e)
    dd=(running-e)/np.maximum(running,1e-12)
    daily=np.diff(e)/np.maximum(e[:-1],1e-12) if len(e)>1 else np.array([])
    mean_ret=float(daily.mean()) if len(daily) else 0.0
    std_ret=float(daily.std(ddof=1)) if len(daily)>1 else 0.0
    sharpe=mean_ret/std_ret*math.sqrt(periods_per_year) if std_ret>0 else 0.0
    downside=daily[daily<0]
    down_std=float(downside.std(ddof=1)) if len(downside)>1 else 0.0
    sortino=mean_ret/down_std*math.sqrt(periods_per_year) if down_std>0 else 0.0
    years=max(len(daily)/periods_per_year,1/periods_per_year)
    cagr=(final/initial)**(1/years)-1 if initial>0 else 0.0
    max_dd=float(dd.max()) if len(dd) else 0.0
    wins=[t.pnl for t in trades if t.pnl>0]; losses=[t.pnl for t in trades if t.pnl<=0]
    gross_profit=sum(wins); gross_loss=abs(sum(losses))
    return {
        "initial_capital":initial,"final_equity":final,
        "total_return_pct":(final/initial-1)*100 if initial else 0.0,
        "cagr_pct":cagr*100,"max_drawdown_pct":max_dd*100,
        "max_drawdown_duration_bars":int(_max_drawdown_duration(e)),
        "total_trades":len(trades),
        "win_rate_pct":len(wins)/len(trades)*100 if trades else 0.0,
        "profit_factor":gross_profit/gross_loss if gross_loss else float("inf"),
        "expectancy_per_trade":float(np.mean([t.pnl for t in trades])) if trades else 0.0,
        "sharpe":sharpe,"sortino":sortino,
        "calmar":cagr/max_dd if max_dd else float("inf"),
        "avg_win":gross_profit/len(wins) if wins else 0.0,
        "avg_loss":-gross_loss/len(losses) if losses else 0.0,
    }


def _max_drawdown_duration(equity:np.ndarray)->int:
    peak=equity[0]; duration=0; longest=0
    for x in equity:
        if x>=peak: peak=x; duration=0
        else: duration+=1; longest=max(longest,duration)
    return longest
