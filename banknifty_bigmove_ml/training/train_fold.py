"""
train_fold.py
--------------
Trains one model config on one walk-forward fold, evaluates on that fold's
validation block, and returns FoldMetrics. Pure function — no MLflow calls
here, so it's independently testable; run_experiment.py owns all logging.

Works for both a single-timeframe model (X_train/X_val are plain ndarrays,
model(xb) takes a tensor) and a MultiTimeframeModel (X_train/X_val are
{tf_name: ndarray} dicts, model(xb) takes a {tf_name: tensor} dict) — the
batching helpers below dispatch on which type they were given.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from metrics import compute_fold_metrics, FoldMetrics, compute_directional_strategy_metrics, DirectionalStrategyMetrics
from models import build_model


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _n_samples(X):
    return next(iter(X.values())).shape[0] if isinstance(X, dict) else X.shape[0]


def _to_tensor(X):
    if isinstance(X, dict):
        return {tf: torch.from_numpy(arr).float() for tf, arr in X.items()}
    return torch.from_numpy(X).float()


def _index_batch(X_t, idx):
    if isinstance(X_t, dict):
        return {tf: t[idx] for tf, t in X_t.items()}
    return X_t[idx]


def _to_device(X_t, device):
    if isinstance(X_t, dict):
        return {tf: t.to(device) for tf, t in X_t.items()}
    return X_t.to(device)


def train_and_eval_fold(
    model_name: str,
    model_kwargs: dict,
    X_train, y_train: np.ndarray,
    X_val, y_val: np.ndarray,
    epochs: int, lr: float, batch_size: int, seed: int,
    device: str = "cpu",
) -> tuple[FoldMetrics, dict, nn.Module]:
    set_seed(seed)
    model = build_model(model_name, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    X_train_t = _to_tensor(X_train)
    y_train_t = torch.from_numpy(y_train).float()
    n_train = _n_samples(X_train)

    model.train()
    history = {"train_loss": []}
    for epoch in range(epochs):
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb = _to_device(_index_batch(X_train_t, idx), device)
            yb = y_train_t[idx].to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        history["train_loss"].append(epoch_loss / n_train)

    model.eval()
    X_val_t = _to_tensor(X_val)  # kept on CPU; moved to device per-batch below
    n_val = _n_samples(X_val)
    val_probs_chunks = []
    with torch.no_grad():
        # Batched, same as training — an unbatched full-validation-set forward pass
        # is what caused a real 28GB CUDA OOM for the Transformer at lookback=240
        # (self-attention's memory cost grows with batch size x sequence length^2;
        # materializing the whole val set at once is never safe to assume fits).
        for start in range(0, n_val, batch_size):
            idx = torch.arange(start, min(start + batch_size, n_val))
            xb = _to_device(_index_batch(X_val_t, idx), device)
            val_probs_chunks.append(torch.sigmoid(model(xb)).cpu().numpy())
    val_probs = np.concatenate(val_probs_chunks)

    fold_metrics = compute_fold_metrics(y_val, val_probs)
    return fold_metrics, history, model


def train_and_eval_fold_directional(
    model_name: str,
    model_kwargs: dict,
    X_train, y_train: np.ndarray,   # y_*: (n, 2) — [long_wins, short_wins]
    X_val, y_val: np.ndarray,
    epochs: int, lr: float, batch_size: int, seed: int,
    device: str = "cpu", strategy_threshold: float = 0.5,
) -> tuple[FoldMetrics, FoldMetrics, DirectionalStrategyMetrics, dict, nn.Module]:
    """Directional (triple-barrier) counterpart to train_and_eval_fold —
    kept as a SEPARATE function (duplicating the ~25-line training loop)
    rather than branching inside the original, so the already-registered
    v1/v2 single-target models and their exact training code path are never
    touched by this addition. model_kwargs must include n_targets=2 (or the
    caller passes it directly to build_model via model_kwargs).

    Returns (long_metrics, short_metrics, strategy_metrics, history, model).
    """
    set_seed(seed)
    model = build_model(model_name, **{**model_kwargs, "n_targets": 2}).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    X_train_t = _to_tensor(X_train)
    y_train_t = torch.from_numpy(y_train).float()
    n_train = _n_samples(X_train)

    model.train()
    history = {"train_loss": []}
    for epoch in range(epochs):
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb = _to_device(_index_batch(X_train_t, idx), device)
            yb = y_train_t[idx].to(device)  # (batch, 2)

            optimizer.zero_grad()
            logits = model(xb)  # (batch, 2)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        history["train_loss"].append(epoch_loss / n_train)

    model.eval()
    X_val_t = _to_tensor(X_val)
    n_val = _n_samples(X_val)
    val_probs_chunks = []
    with torch.no_grad():
        for start in range(0, n_val, batch_size):
            idx = torch.arange(start, min(start + batch_size, n_val))
            xb = _to_device(_index_batch(X_val_t, idx), device)
            val_probs_chunks.append(torch.sigmoid(model(xb)).cpu().numpy())
    val_probs = np.concatenate(val_probs_chunks)  # (n_val, 2)

    long_metrics = compute_fold_metrics(y_val[:, 0], val_probs[:, 0])
    short_metrics = compute_fold_metrics(y_val[:, 1], val_probs[:, 1])
    strategy_metrics = compute_directional_strategy_metrics(
        p_long=val_probs[:, 0], p_short=val_probs[:, 1],
        long_wins=y_val[:, 0], short_wins=y_val[:, 1],
        threshold=strategy_threshold,
    )
    return long_metrics, short_metrics, strategy_metrics, history, model
