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


def fit_with_early_stopping(model, X_train, y_train, X_val, y_val):
    kwargs = {"eval_set": [(X_val, y_val)]}
    if hasattr(model, "fit"):
        try:
            model.fit(
                X_train,
                y_train,
                callbacks=[],
                **kwargs,
            )
        except TypeError:
            model.fit(X_train, y_train, **kwargs)
    return model


def align_classes(probabilities: np.ndarray, classes: np.ndarray, class_order=(-1, 0, 1)) -> np.ndarray:
    out = np.zeros((len(probabilities), len(class_order)), dtype=float)
    for i, cls in enumerate(class_order):
        matches = np.flatnonzero(classes == cls)
        if matches.size:
            out[:, i] = probabilities[:, matches[0]]
    row_sum = out.sum(axis=1, keepdims=True)
    return out / np.where(row_sum == 0, 1.0, row_sum)


def predict_probabilities(model, X: np.ndarray, *, class_order=(-1, 0, 1)) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X))
    classes = np.asarray(getattr(model, "classes_", np.arange(probabilities.shape[1]) - 1))
    if probabilities.shape[1] == len(class_order) and set(classes.tolist()) == set(class_order):
        return align_classes(probabilities, classes, class_order)
    return probabilities
