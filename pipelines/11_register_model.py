from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow


def run() -> None:
    if os.getenv("ALLOW_MODEL_REGISTRATION") != "1":
        raise SystemExit(
            "Registration is intentionally disabled. Set ALLOW_MODEL_REGISTRATION=1 "
            "only after the research/holdout gates are approved."
        )
    report = json.loads(Path("artifacts/validation/report.json").read_text())
    if not report.get("holdout_evaluated"):
        raise SystemExit("Refusing registration without a frozen-holdout evaluation")

    experiment = mlflow.get_experiment_by_name("BankNifty_Research")
    if experiment is None:
        raise SystemExit("MLflow experiment not found")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise SystemExit("No MLflow run found")

    run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/artifacts/models/lgbm_dev.joblib"
    registered = mlflow.register_model(model_uri, "BankNifty_LGBM")
    print(f"Registered model version {registered.version}")


if __name__ == "__main__":
    run()
