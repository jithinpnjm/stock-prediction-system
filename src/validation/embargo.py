from __future__ import annotations

import numpy as np


def embargo_indices(
    event_start_ns: np.ndarray,
    *,
    boundary_ns: int,
    duration_ns: int,
) -> np.ndarray:
    return np.flatnonzero(
        (event_start_ns >= boundary_ns)
        & (event_start_ns <= boundary_ns + duration_ns)
    )
