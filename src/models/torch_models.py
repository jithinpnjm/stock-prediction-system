from __future__ import annotations

import math

import torch
from torch import nn


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        padding = (3 - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = y[..., : x.size(-1)]
        y = self.dropout(self.activation(y))
        y = self.conv2(y)
        y = y[..., : x.size(-1)]
        return self.activation(x + self.dropout(y))


class TCNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        levels: int = 4,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [TCNBlock(hidden_dim, 2**i, dropout) for i in range(levels)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = self.input(y)
        for block in self.blocks:
            y = block(y)
        y = self.norm(y[:, :, -1])
        return self.head(y)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 3,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input = nn.Linear(input_dim, d_model)
        self.position = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.position(self.input(x))
        seq_len = y.size(1)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=y.device),
            diagonal=1,
        )
        y = self.encoder(y, mask=mask)
        return self.head(self.norm(y[:, -1]))
