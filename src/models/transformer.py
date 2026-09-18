from __future__ import annotations

import torch
from torch import nn


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 3,
        n_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj=nn.Linear(n_features,d_model)
        layer=nn.TransformerEncoderLayer(
            d_model=d_model,nhead=n_heads,dim_feedforward=4*d_model,
            dropout=dropout,batch_first=True,norm_first=True,activation="gelu"
        )
        self.encoder=nn.TransformerEncoder(layer,num_layers=n_layers)
        self.norm=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,n_classes)

    def forward(self,x):
        z=self.encoder(self.proj(x))
        return self.head(self.norm(z[:,-1,:]))


def build_transformer(n_features: int, **kwargs) -> TemporalTransformer:
    return TemporalTransformer(n_features, **kwargs)
