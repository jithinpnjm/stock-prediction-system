from __future__ import annotations

import numpy as np


def apply_embargo(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    timestamps: np.ndarray,
    embargo: int,
) -> np.ndarray:
    if embargo <= 0 or len(train_idx) == 0 or len(test_idx) == 0:
        return train_idx
    cutoff = np.asarray(timestamps)[test_idx].min() - embargo
    return train_idx[np.asarray(timestamps)[train_idx] < cutoff]
