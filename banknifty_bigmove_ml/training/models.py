"""
models.py
----------
Candidate architectures over raw OHLCV sequence windows (Phase 3). All
input is (batch, lookback, N_FEATURES) — no hand-crafted indicators, per
project requirement (N_FEATURES channels are lossless OHLC-derived shape
features — see windowing.py). Deliberately modest scale: the GPU buys
iteration speed across many sweep trials, not model size.

Each architecture exposes `.encode(x) -> (batch, feature_dim)` (the pooled
representation before the classification head) in addition to `.forward(x)`
(encode + head, for standalone single-timeframe use). MultiTimeframeModel
reuses `.encode()` from one encoder instance per timeframe branch, so a
1-min + 5-min fusion model is just two of these encoders concatenated
before one shared head — no separate multi-timeframe architecture needed.

n_targets: 1 (original ±0.3%-any-direction label, output shape (batch,)) or
2 (directional triple-barrier label — [long_wins, short_wins] logits,
output shape (batch, 2)) — added 2026-09-13 without touching the n_targets=1
path, so the already-registered v1/v2 models stay exactly reproducible.
"""
import torch
import torch.nn as nn

from windowing import N_FEATURES


def _head_output(head_out: torch.Tensor, n_targets: int) -> torch.Tensor:
    return head_out.squeeze(-1) if n_targets == 1 else head_out


class TCN(nn.Module):
    """Dilated 1D-conv stack — good inductive bias for local price-pattern
    detection at low compute cost."""

    def __init__(self, channels=(32, 32, 32), kernel_size=5, dropout=0.1, n_features=N_FEATURES, n_targets=1):
        super().__init__()
        layers = []
        in_ch = n_features
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation  # causal: pad left only, handled via slicing below
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.conv_layers = nn.ModuleList(layers)
        self.feature_dim = in_ch
        self.n_targets = n_targets
        self.head = nn.Linear(in_ch, n_targets)

    def encode(self, x):  # x: (batch, lookback, n_features) -> (batch, feature_dim)
        x = x.transpose(1, 2)  # (batch, n_features, lookback)
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv1d):
                x = layer(x)
                x = x[:, :, :-layer.padding[0]] if layer.padding[0] > 0 else x  # causal trim
            else:
                x = layer(x)
        return x.mean(dim=-1)  # global average pool over time

    def forward(self, x):
        return _head_output(self.head(self.encode(x)), self.n_targets)


class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.1, n_features=N_FEATURES, n_targets=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.feature_dim = hidden_size
        self.n_targets = n_targets
        self.head = nn.Linear(hidden_size, n_targets)

    def encode(self, x):  # x: (batch, lookback, n_features) -> (batch, feature_dim)
        _out, (h_n, _c) = self.lstm(x)
        return h_n[-1]

    def forward(self, x):
        return _head_output(self.head(self.encode(x)), self.n_targets)


class SmallTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, dropout=0.1, max_len=512, n_features=N_FEATURES, n_targets=1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.feature_dim = d_model
        self.n_targets = n_targets
        self.head = nn.Linear(d_model, n_targets)

    def encode(self, x):  # x: (batch, lookback, n_features) -> (batch, feature_dim)
        seq_len = x.shape[1]
        h = self.input_proj(x) + self.pos_embedding[:, :seq_len, :]
        h = self.encoder(h)
        return h.mean(dim=1)  # mean pool over time

    def forward(self, x):
        return _head_output(self.head(self.encode(x)), self.n_targets)


MODEL_REGISTRY = {
    "tcn": TCN,
    "lstm": LSTMClassifier,
    "transformer": SmallTransformer,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name == "mtf":
        return MultiTimeframeModel(**kwargs)
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}' — choices: {list(MODEL_REGISTRY)} or 'mtf'")
    return MODEL_REGISTRY[name](**kwargs)


class MultiTimeframeModel(nn.Module):
    """Fuses one encoder branch per timeframe (e.g. 1-min + 5-min): each
    branch encodes its own window independently, the pooled representations
    are concatenated, and one shared head produces the final logit(s).
    Branch encoders can be different architectures/sizes per timeframe."""

    def __init__(self, branch_specs: dict, n_targets: int = 1):
        # branch_specs: {"1min": ("tcn", {...kwargs}), "5min": ("lstm", {...kwargs})}
        super().__init__()
        self.tf_names = list(branch_specs.keys())
        self.branches = nn.ModuleDict({
            tf: build_model(arch_name, **kwargs) for tf, (arch_name, kwargs) in branch_specs.items()
        })
        total_dim = sum(branch.feature_dim for branch in self.branches.values())
        self.n_targets = n_targets
        self.head = nn.Linear(total_dim, n_targets)

    def forward(self, x_by_tf: dict):
        # x_by_tf: {"1min": tensor(batch, lookback_1m, n_features), "5min": tensor(batch, lookback_htf, n_features)}
        encoded = [self.branches[tf].encode(x_by_tf[tf]) for tf in self.tf_names]
        fused = torch.cat(encoded, dim=-1)
        return _head_output(self.head(fused), self.n_targets)
