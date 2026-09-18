from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def train_meta_model(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingClassifier:
    model=HistGradientBoostingClassifier(
        max_iter=300,learning_rate=0.05,max_leaf_nodes=15,l2_regularization=1.0,
        random_state=42
    )
    model.fit(X,y)
    return model


def meta_probability(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:,1]
