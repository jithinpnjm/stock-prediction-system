import numpy as np

from src.features.fractional_diff import frac_diff


def test_fractional_difference_returns_finite_later_values():
    x=np.linspace(100,120,100)
    out=frac_diff(x,0.4,max_lags=20)
    assert np.isfinite(out[20:]).all()
