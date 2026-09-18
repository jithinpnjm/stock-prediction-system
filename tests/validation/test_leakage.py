import numpy as np
import pytest

from src.validation.leakage import assert_feature_availability,assert_no_overlap


def test_feature_future_time_is_rejected():
    with pytest.raises(ValueError):
        assert_feature_availability(np.array([2]),np.array([1]))


def test_overlapping_intervals_are_rejected():
    with pytest.raises(ValueError):
        assert_no_overlap(np.array([0]),np.array([5]),np.array([4]),np.array([7]))
