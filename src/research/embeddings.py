from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA

def fit_pca_embedding(X:np.ndarray,n_components:int=16):
    n_components=min(n_components,X.shape[0],X.shape[1])
    return PCA(n_components=n_components,random_state=42).fit(X)

def transform_embedding(model,X:np.ndarray)->np.ndarray:
    return model.transform(X)
