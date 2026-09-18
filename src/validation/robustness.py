from __future__ import annotations

import numpy as np

from .statistical_tests import sharpe_ratio


def subperiod_returns(
    timestamps,
    returns: np.ndarray,
) -> list[dict[str, float | str]]:
    ts = np.asarray(timestamps)
    r = np.asarray(returns, dtype=float)
    if len(ts) != len(r):
        raise ValueError("timestamps and returns must have the same length")
    dates = ts.astype("datetime64[D]")
    out=[]
    for d in np.unique(dates):
        x=r[dates == d]
        out.append({
            "period": str(d),
            "mean_return": float(x.mean()) if len(x) else 0.0,
            "sharpe": sharpe_ratio(x, periods_per_year=252),
            "observations": float(len(x)),
        })
    return out


def block_bootstrap(
    returns: np.ndarray,
    block_size: int = 20,
    n_paths: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    r=np.asarray(returns, dtype=float)
    rng=np.random.default_rng(seed)
    if r.size == 0:
        return np.empty((0,0))
    n_blocks=int(np.ceil(len(r)/block_size))
    paths=np.empty((n_paths, len(r)))
    for i in range(n_paths):
        starts=rng.integers(0, max(len(r)-block_size+1,1), size=n_blocks)
        sample=np.concatenate([r[s:s+block_size] for s in starts])[:len(r)]
        paths[i]=sample
    return paths
