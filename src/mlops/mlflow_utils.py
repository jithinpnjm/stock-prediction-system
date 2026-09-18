from __future__ import annotations

import json
from pathlib import Path

import mlflow


def start_experiment(name: str):
    mlflow.set_experiment(name)
    return mlflow.start_run()


def log_lineage(lineage: dict[str, str]) -> None:
    safe_params = {str(k): str(v)[:500] for k, v in lineage.items()}
    mlflow.log_params(safe_params)


def log_json_artifact(payload: dict, name: str) -> None:
    path = Path("artifacts") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    mlflow.log_artifact(str(path))


def log_metrics(metrics: dict[str, float]) -> None:
    mlflow.log_metrics(
        {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and v == v
        }
    )
