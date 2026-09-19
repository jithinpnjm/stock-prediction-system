"""Self-supervised next-bar-return pretraining for TemporalTransformer.

The pretraining target for the window ending at bar t is the ATR-
normalized return realized from bar t to bar t+1 -- computed the same
way the triple-barrier label is computed (from bars strictly after the
window), so this carries no more leakage risk than the labels already
used elsewhere in this codebase. The window itself never includes bar
t+1, only bars up to and including t.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class PretrainResult:
    model: nn.Module
    best_epoch: int
    train_loss: float
    val_loss: float


def pretrain_encoder(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 30,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    patience: int = 6,
    device: str | None = None,
) -> PretrainResult:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    model = model.to(dev)

    train_ds = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.as_tensor(X_val, dtype=torch.float32), torch.as_tensor(y_val, dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    train_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            optimizer.zero_grad(set_to_none=True)
            pred = model.forward_pretrain(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(yb)
            count += len(yb)
        train_loss = total / max(count, 1)

        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(dev)
                yb = yb.to(dev)
                pred = model.forward_pretrain(xb)
                loss = loss_fn(pred, yb)
                total += float(loss.item()) * len(yb)
                count += len(yb)
        val_loss = total / max(count, 1)

        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_epoch = epoch
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return PretrainResult(model, best_epoch, train_loss, best_loss)
