"""
inference.py
-------------
Loads a registered model from the MLflow Model Registry and turns a rolling
buffer of raw 1-min OHLCV candles into P(big move in the next 45 minutes).

This is the ONLY place live/replay code should build a model input window —
it reuses windowing.py's exact channel construction (candle-shape features,
anchor-relative scaling) so there is zero chance of train/serve skew between
what the model was trained on and what it sees at inference time.
"""
from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
from windowing import _ohlc_shape_channels, N_FEATURES  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlflow_config import set_tracking  # noqa: E402

import mlflow.pytorch  # noqa: E402


@dataclass
class Prediction:
    datetime: pd.Timestamp
    close: float
    probability: float   # P(big move within next 45 candles), per the trained label definition


class BigMoveModel:
    """Wraps a registered model + the rolling window buffer it needs.
    One instance per (model, lookback) — feed it closed 1-min candles one
    at a time via `on_new_candle`, get a Prediction back once the buffer is full."""

    def __init__(self, registered_name: str, stage_or_version: str = "latest", lookback: int = 120, device: str = "cpu"):
        set_tracking()
        model_uri = f"models:/{registered_name}/{stage_or_version}"
        self.model = mlflow.pytorch.load_model(model_uri, map_location=device)
        self.model.eval()
        self.device = device
        self.lookback = lookback
        self.buffer = deque(maxlen=lookback)   # each item: (open, high, low, close)

    def on_new_candle(self, datetime_, open_, high_, low_, close_) -> Prediction | None:
        """Feed one CLOSED candle. Returns a Prediction once the buffer has
        `lookback` candles, else None (not enough history yet)."""
        self.buffer.append((open_, high_, low_, close_))
        if len(self.buffer) < self.lookback:
            return None

        arr = np.array(self.buffer, dtype=np.float64)   # (lookback, 4)
        o, h, l, c = arr[:, 0][None, :], arr[:, 1][None, :], arr[:, 2][None, :], arr[:, 3][None, :]
        anchor = np.array([close_], dtype=np.float64)    # anchor = the just-closed candle's own close

        X = _ohlc_shape_channels(o, h, l, c, anchor)      # (1, lookback, N_FEATURES) — identical to training
        assert X.shape == (1, self.lookback, N_FEATURES)

        with torch.no_grad():
            logit = self.model(torch.from_numpy(X).float().to(self.device))
            prob = torch.sigmoid(logit).cpu().numpy()[0]

        return Prediction(datetime=pd.Timestamp(datetime_), close=float(close_), probability=float(prob))

    def reset(self):
        self.buffer.clear()


def replay_dataframe(model: BigMoveModel, df: pd.DataFrame) -> pd.DataFrame:
    """Feed an entire historical dataframe (columns: datetime,open,high,low,close)
    through the model candle-by-candle, exactly as live inference would see it.
    Used by simulate.py for backtesting against the untouched holdout."""
    df = df.sort_values("datetime").reset_index(drop=True)
    rows = []
    for row in df.itertuples(index=False):
        pred = model.on_new_candle(row.datetime, row.open, row.high, row.low, row.close)
        if pred is not None:
            rows.append({"datetime": pred.datetime, "close": pred.close, "probability": pred.probability})
    return pd.DataFrame(rows)
