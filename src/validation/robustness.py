from __future__ import annotations

from itertools import combinations
import math
import numpy as np

from .statistical_tests import sharpe_ratio


def subperiod_returns(timestamps,returns:np.ndarray):
    ts=np.asarray(timestamps); r=np.asarray(returns,dtype=float)
    if len(ts)!=len(r):raise ValueError("timestamps and returns must have equal length")
    dates=ts.astype("datetime64[D]"); out=[]
    for d in np.unique(dates):
        x=r[dates==d]
        out.append({
            "period":str(d),"mean_return":float(x.mean()) if len(x) else 0.0,
            "sharpe":sharpe_ratio(x),"observations":int(len(x))
        })
    return out


def block_bootstrap(returns:np.ndarray,block_size:int=20,n_paths:int=1000,seed:int=42):
    r=np.asarray(returns,dtype=float)
    if r.size==0:return np.empty((0,0))
    rng=np.random.default_rng(seed); paths=np.empty((n_paths,len(r)))
    n_blocks=int(math.ceil(len(r)/block_size))
    for i in range(n_paths):
        starts=rng.integers(0,max(len(r)-block_size+1,1),size=n_blocks)
        paths[i]=np.concatenate([r[s:s+block_size] for s in starts])[:len(r)]
    return paths


def probability_of_backtest_overfitting(
    strategy_returns:np.ndarray,
    n_partitions:int=8,
)->float:
    """CSC-style PBO diagnostic using in-sample Sharpe rank vs OOS rank.

    This is a compact diagnostic; full production inference should retain all
    strategy paths and trial metadata rather than relying on this scalar alone.
    """
    x=np.asarray(strategy_returns,dtype=float)
    if x.ndim!=2 or x.shape[0]<2 or x.shape[1]<n_partitions:
        raise ValueError("strategy_returns must be [observations, strategies]")
    groups=np.array_split(np.arange(x.shape[0]),n_partitions)
    half=n_partitions//2
    if half<2:return float("nan")
    failures=0; total=0
    for test_groups in combinations(range(n_partitions),half):
        is_idx=np.concatenate([groups[i] for i in test_groups])
        os_idx=np.concatenate([groups[i] for i in range(n_partitions) if i not in test_groups])
        is_scores=np.nanmean(x[is_idx],axis=0)/np.nanstd(x[is_idx],axis=0)
        os_scores=np.nanmean(x[os_idx],axis=0)/np.nanstd(x[os_idx],axis=0)
        best=int(np.nanargmax(is_scores))
        os_rank=np.mean(os_scores<=os_scores[best])
        failures+=int(os_rank<0.5); total+=1
    return failures/total if total else float("nan")
