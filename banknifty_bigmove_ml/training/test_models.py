import torch

from models import build_model, MODEL_REGISTRY, MultiTimeframeModel
from windowing import N_FEATURES


def test_all_models_forward_pass_shapes():
    batch, lookback = 8, 60
    x = torch.randn(batch, lookback, N_FEATURES)
    for name in MODEL_REGISTRY:
        model = build_model(name)
        out = model(x)
        assert out.shape == (batch,), f"{name} produced shape {out.shape}, expected ({batch},)"


def test_tcn_causality_output_finite_for_short_and_long_lookback():
    for lookback in (10, 60, 240):
        x = torch.randn(4, lookback, N_FEATURES)
        model = build_model("tcn")
        out = model(x)
        assert torch.isfinite(out).all()


def test_multi_timeframe_model_fuses_branches():
    batch = 6
    x_by_tf = {
        "1min": torch.randn(batch, 120, N_FEATURES),
        "5min": torch.randn(batch, 24, N_FEATURES),
    }
    model = build_model("mtf", branch_specs={"1min": ("tcn", {}), "5min": ("lstm", {})})
    out = model(x_by_tf)
    assert out.shape == (batch,)
    assert torch.isfinite(out).all()
    # feature_dim should be the sum of both branches' encoder output dims
    assert model.head.in_features == model.branches["1min"].feature_dim + model.branches["5min"].feature_dim


def test_n_targets_2_produces_dual_directional_output():
    """The directional (triple-barrier) label needs [long_wins, short_wins]
    logits per sample — output shape (batch, 2), not squeezed to (batch,)."""
    batch, lookback = 8, 60
    x = torch.randn(batch, lookback, N_FEATURES)
    for name in MODEL_REGISTRY:
        model = build_model(name, n_targets=2)
        out = model(x)
        assert out.shape == (batch, 2), f"{name} n_targets=2 produced {out.shape}"
        assert torch.isfinite(out).all()


def test_n_targets_1_default_unchanged_shape():
    """Backward-compat guard: existing registered v1/v2 models rely on the
    default n_targets=1 squeezing to (batch,) — must never silently change."""
    batch, lookback = 8, 60
    x = torch.randn(batch, lookback, N_FEATURES)
    for name in MODEL_REGISTRY:
        model = build_model(name)
        assert model.n_targets == 1
        out = model(x)
        assert out.shape == (batch,)


def test_multi_timeframe_model_dual_target():
    batch = 6
    x_by_tf = {
        "1min": torch.randn(batch, 120, N_FEATURES),
        "5min": torch.randn(batch, 24, N_FEATURES),
    }
    model = build_model("mtf", branch_specs={"1min": ("tcn", {}), "5min": ("lstm", {})}, n_targets=2)
    out = model(x_by_tf)
    assert out.shape == (batch, 2)
    assert torch.isfinite(out).all()
