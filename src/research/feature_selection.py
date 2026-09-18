from __future__ import annotations
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def mutual_information_ranking(X:np.ndarray,y:np.ndarray,feature_names:list[str]):
    scores=mutual_info_classif(X,y,random_state=42)
    return sorted(zip(feature_names,scores.tolist()),key=lambda x:x[1],reverse=True)
