from __future__ import annotations

import math

import numpy as np
from scipy import stats


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return 0.0, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def permutation_p_value(
    values: np.ndarray,
    n_perm: int = 5000,
    seed: int = 42,
) -> float:
    x = np.asarray(values, dtype=float)
    observed = abs(x.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    centered = x - x.mean()
    for _ in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(x))
        exceed += abs(np.mean(centered * signs)) >= observed
    return float((exceed + 1) / (n_perm + 1))


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods_per_year))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Approximate DSR-style exceedance probability.

    This is an audit diagnostic, not a replacement for a formal inference
    package. It adjusts the expected null Sharpe for the search count.
    """
    if n_trials <= 1 or n_observations <= 1:
        return float(stats.norm.cdf(observed_sharpe))
    expected_max = math.sqrt(2 * math.log(max(n_trials, 2)))
    variance = max(
        1.0
        - skew * observed_sharpe
        + ((kurtosis - 1.0) / 4.0) * observed_sharpe**2,
        1e-9,
    )
    z = (observed_sharpe - expected_max) / math.sqrt(variance / n_observations)
    return float(stats.norm.cdf(z))
