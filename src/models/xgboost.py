from __future__ import annotations

import numpy as np
import xgboost as xgb


def train_xgboost_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="mlogloss",
        random_state=seed,
        tree_method="hist",
    )
    model.fit(X, np.asarray(y, dtype=int) + 1, verbose=False)
    return model


def predict_proba(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)
