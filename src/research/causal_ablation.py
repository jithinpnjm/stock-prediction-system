from __future__ import annotations

import numpy as np
import pandas as pd


def ablation_columns(
    X: pd.DataFrame,
    columns: list[str],
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for column in columns:
        if column in X.columns:
            result[column] = X.drop(columns=[column])
    return result


def shuffled_feature_importance(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    feature: str,
    *,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    original = model.predict_proba(X)[:, -1]
    shuffled = X.copy()
    shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
    shuffled_prob = model.predict_proba(shuffled)[:, -1]
    return float(np.mean(original * (1 - original)) - np.mean(shuffled_prob * (1 - shuffled_prob)))
