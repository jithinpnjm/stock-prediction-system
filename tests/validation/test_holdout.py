import numpy as np
import pytest

from src.validation.holdout import chronological_holdout,verify_holdout_is_future


def test_holdout_is_strictly_future():
    ts=np.arange(100)
    dev,holdout=chronological_holdout(ts,holdout_fraction=0.2)
    verify_holdout_is_future(ts,dev,holdout)
    assert ts[dev].max()<ts[holdout].min()


def test_holdout_overlap_fails():
    ts=np.arange(10)
    with pytest.raises(ValueError):
        verify_holdout_is_future(ts,np.arange(0,8),np.arange(7,10))
