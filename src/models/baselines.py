from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic_regression(*, c: float = 1.0, seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c,
                    max_iter=2000,
                    random_state=seed,
                    class_weight="balanced",
                    multi_class="auto",
                ),
            ),
        ]
    )


def make_lightgbm_classifier(
    *, seed: int = 42, params: dict[str, Any] | None = None
) -> lgb.LGBMClassifier:
    base = {
        "objective": "multiclass",
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "class_weight": "balanced",
        "random_state": seed,
        "verbosity": -1,
        "n_jobs": -1,
    }
    base.update(params or {})
    return lgb.LGBMClassifier(**base)


def make_xgboost_classifier(*, seed: int = 42, params: dict[str, Any] | None = None):
    from xgboost import XGBClassifier

    base = {
        "objective": "multi:softprob",
        "num_class": 3,
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": -1,
    }
    base.update(params or {})
    return XGBClassifier(**base)


def predict_probabilities(model, X: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X))
    if probabilities.shape[1] != 3:
        raise ValueError("Expected a 3-class probability output")
    return probabilities
