from __future__ import annotations

import torch
from torch import nn


class CausalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        y = self.conv(x)
        y = y[..., : x.shape[-1]]
        return self.dropout(self.act(self.norm(y))) + self.residual(x)


class TCNClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        channels=(64, 64, 32),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        c = n_features
        for i, out in enumerate(channels):
            layers.append(CausalConvBlock(c, out, kernel_size, 2**i, dropout))
            c = out
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(c, n_classes))

    def forward(self, x):
        return self.head(self.encoder(x))


def build_tcn(n_features: int, n_classes: int = 3, **kwargs) -> TCNClassifier:
    return TCNClassifier(n_features, n_classes, **kwargs)
