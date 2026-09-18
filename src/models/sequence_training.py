from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def train_sequence_classifier(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, *, epochs: int = 50, batch_size: int = 256, learning_rate: float = 1e-3, weight_decay: float = 1e-4, patience: int = 8, seed: int = 42, device: str | None = None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device_obj)
    train_x = torch.as_tensor(X_train, dtype=torch.float32)
    train_y = torch.as_tensor(y_train, dtype=torch.long)
    val_x = torch.as_tensor(X_val, dtype=torch.float32)
    val_y = torch.as_tensor(y_val, dtype=torch.long)
    if train_x.ndim != 3:
        raise ValueError("Sequence input must have shape [samples, window, features]")
    if len(train_x) != len(train_y) or len(val_x) != len(val_y):
        raise ValueError("Feature/label lengths differ")
    counts = np.bincount(train_y.numpy(), minlength=3).astype(float)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device_obj))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=False)
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device_obj), yb.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_x.to(device_obj)), val_y.to(device_obj)).detach().cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, history
