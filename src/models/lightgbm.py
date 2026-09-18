from typing import Any

import lightgbm as lgb
import numpy as np


def train_lightgbm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any] = None,
) -> lgb.Booster:
    """
    Trains a LightGBM classifier.
    """
    if params is None:
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 5,
            "feature_fraction": 0.8,
            # "device": "gpu", # Disabled temporarily as native pip wheels don't pack CMake GPU by default.
            "seed": 42,
            "verbose": -1,
        }

    # Map labels [-1, 0, 1] to [0, 1, 2] for LightGBM multiclass
    y_train_mapped = y_train + 1
    y_val_mapped = y_val + 1

    train_data = lgb.Dataset(X_train, label=y_train_mapped)
    val_data = lgb.Dataset(X_val, label=y_val_mapped, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
    )

    return model


def predict_lightgbm(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """
    Predicts classes. Re-maps [0, 1, 2] back to [-1, 0, 1].
    """
    probs = model.predict(X)
    classes = np.argmax(probs, axis=1)
    return classes - 1
