import torch

from src.models.torch_models import TCNClassifier, TransformerClassifier


def test_tcn_shape_and_causality_path():
    model = TCNClassifier(input_dim=4, hidden_dim=8, levels=2)
    x = torch.randn(2, 16, 4)
    y = model(x)
    assert y.shape == (2, 3)


def test_transformer_shape():
    model = TransformerClassifier(input_dim=4, d_model=16, nhead=4, layers=2)
    x = torch.randn(2, 16, 4)
    y = model(x)
    assert y.shape == (2, 3)
