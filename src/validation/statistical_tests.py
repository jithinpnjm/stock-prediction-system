from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, values.size), replace=True)
    means = samples.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def permutation_pvalue(
    values: np.ndarray,
    *,
    null_value: float = 0.0,
    n_permutations: int = 2000,
    seed: int = 42,
) -> float:
    values = np.asarray(values, dtype=float) - null_value
    observed = abs(values.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, values.size))
    permuted = abs((signs * values).mean(axis=1))
    return float((np.sum(permuted >= observed) + 1) / (n_permutations + 1))


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 19_500) -> float:
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns, ddof=1) * periods_per_year**0.5)
