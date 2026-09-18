from __future__ import annotations

import numpy as np


def assert_feature_availability(
    feature_timestamps: np.ndarray,
    event_timestamps: np.ndarray,
) -> None:
    feature_ts = np.asarray(feature_timestamps)
    event_ts = np.asarray(event_timestamps)
    if feature_ts.shape != event_ts.shape:
        raise ValueError("feature and event timestamps must have equal shape")
    if np.any(feature_ts > event_ts):
        bad = int(np.sum(feature_ts > event_ts))
        raise ValueError(f"future feature availability detected in {bad} rows")


def assert_no_overlap(
    train_start: np.ndarray,
    train_end: np.ndarray,
    test_start: np.ndarray,
    test_end: np.ndarray,
) -> None:
    a = np.asarray(train_start)
    b = np.asarray(train_end)
    for start, end in zip(np.asarray(test_start), np.asarray(test_end), strict=False):
        if np.any((a <= end) & (b >= start)):
            raise ValueError("overlapping train/test event intervals detected")
