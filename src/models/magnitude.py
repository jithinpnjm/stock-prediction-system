from __future__ import annotations

import lightgbm as lgb


def make_mfe_regressor(*, seed: int = 42, params: dict | None = None):
    base = {
        "objective": "regression",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "random_state": seed,
        "verbosity": -1,
    }
    base.update(params or {})
    return lgb.LGBMRegressor(**base)
