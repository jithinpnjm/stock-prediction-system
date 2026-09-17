"""
Shared MLflow constants. Import this instead of hardcoding the tracking URI
or experiment name anywhere else, so every script (local labeling checks,
training runs on the Nebius VM, future agent backtests) points at the same
place.

Local dev (no Nebius connection): tracking URI falls back to a local
./mlruns directory so labeling/split code can be smoke-tested without a
live MLflow server. Real training runs must override MLFLOW_TRACKING_URI
(e.g. an SSH-tunneled http://localhost:5000 to the Nebius VM).
"""
import os

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = "banknifty_bigmove"


def set_tracking():
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
