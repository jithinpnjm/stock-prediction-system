import torch

from src.models.transformer import build_transformer


def test_classify_and_pretrain_heads_share_backbone_and_shapes():
    model = build_transformer(n_features=5, d_model=16, n_heads=2, n_layers=2)
    x = torch.randn(4, 10, 5)

    class_out = model(x)
    assert class_out.shape == (4, 3)

    pretrain_out = model.forward_pretrain(x)
    assert pretrain_out.shape == (4,)

    # both heads consume the identical pooled encoder representation
    encoded = model.encode(x)
    assert encoded.shape == (4, 16)
