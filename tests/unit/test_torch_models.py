import pytest

torch = pytest.importorskip("torch")

from src.models.tcn import TCNClassifier
from src.models.transformer import TemporalTransformer


def test_tcn_shape_and_causality_path():
    model = TCNClassifier(n_features=4)
    x = torch.randn(2, 4, 16)
    y = model(x)
    assert y.shape == (2, 3)


def test_transformer_shape():
    model = TemporalTransformer(n_features=4, d_model=16, n_heads=4, n_layers=2)
    x = torch.randn(2, 16, 4)
    y = model(x)
    assert y.shape == (2, 3)
