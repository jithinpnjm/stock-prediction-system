from __future__ import annotations

import torch
from torch import nn


class TemporalTransformer(nn.Module):
    """Shared encoder backbone with two interchangeable heads: a
    self-supervised pretraining head (predict next-bar return, a
    continuous scalar) and the supervised triple-barrier classification
    head. Both consume the same pooled representation of a window that
    ends at "now" -- the pretraining target (next bar's return) is
    never part of the input window, exactly the same causal contract
    as the classification label (computed from bars after the window).
    Pretraining lets the encoder learn from every bar in the unlabeled
    5-year history; fine-tuning then only has to adapt a much smaller
    labeled set on top of an already-useful representation.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 3,
        n_classes: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.pretrain_head = nn.Linear(d_model, 1)
        self.max_seq_len = max_seq_len

    def encode(self, x):
        if x.shape[1] > self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        z = self.proj(x) + self.position[:, : x.shape[1], :]
        z = self.encoder(z)
        return self.norm(z[:, -1, :])

    def forward(self, x):
        return self.head(self.encode(x))

    def forward_pretrain(self, x):
        return self.pretrain_head(self.encode(x)).squeeze(-1)


def build_transformer(n_features: int, **kwargs) -> TemporalTransformer:
    return TemporalTransformer(n_features, **kwargs)
