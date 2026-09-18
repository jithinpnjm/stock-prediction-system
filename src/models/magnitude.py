from __future__ import annotations

import numpy as np
import lightgbm as lgb


def train_mfe_regressor(
    X: np.ndarray,
    y_mfe: np.ndarray,
    quantile: float = 0.75,
) -> lgb.LGBMRegressor:
    model=lgb.LGBMRegressor(
        objective="quantile",alpha=quantile,n_estimators=500,
        learning_rate=0.03,num_leaves=31,min_child_samples=30,
        random_state=42,verbosity=-1
    )
    model.fit(X,np.asarray(y_mfe,dtype=float))
    return model
