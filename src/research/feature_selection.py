from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def rank_by_mutual_information(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    X_num = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    valid = X_num.notna().all(axis=1)
    scores = mutual_info_classif(X_num.loc[valid], np.asarray(y)[valid], random_state=seed)
    return (
        pd.DataFrame({"feature": X_num.columns, "mutual_information": scores})
        .sort_values("mutual_information", ascending=False)
        .reset_index(drop=True)
    )
