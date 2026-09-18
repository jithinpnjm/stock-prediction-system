from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np


CLASS_LABELS = np.array([-1, 0, 1], dtype=np.int8)


def train_lightgbm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 1000,
) -> lgb.Booster:
    p = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_data_in_leaf": 30,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
    }
    if params:
        p.update(params)
    train = lgb.Dataset(X_train, label=np.asarray(y_train, dtype=int) + 1)
    valid_sets = [train]
    if X_val is not None and y_val is not None:
        valid_sets.append(
            lgb.Dataset(
                X_val, label=np.asarray(y_val, dtype=int) + 1, reference=train
            )
        )
        callbacks=[lgb.early_stopping(50, verbose=False)]
    else:
        callbacks=[]
    return lgb.train(p, train, num_boost_round=num_boost_round, valid_sets=valid_sets, callbacks=callbacks)


def predict_proba(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


def predict_class(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    return CLASS_LABELS[np.argmax(predict_proba(model, X), axis=1)]
