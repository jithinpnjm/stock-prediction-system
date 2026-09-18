import sys

import mlflow
from mlflow.tracking import MlflowClient


def run():
    print("Running pipeline step: 11_register_model.py")

    mlflow.set_experiment("BankNifty_Baseline_LGBM")
    client = MlflowClient()

    # Get the latest run
    experiment = client.get_experiment_by_name("BankNifty_Baseline_LGBM")
    if not experiment:
        print("ERROR: MLflow experiment not found. Run 07_train.py first.")
        sys.exit(1)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        print("ERROR: No runs found in experiment.")
        sys.exit(1)

    latest_run = runs[0]
    run_id = latest_run.info.run_id
    print(f"Found latest run_id: {run_id}")

    # We will register the fold_1 model as the primary prototype for now
    model_uri = f"runs:/{run_id}/model_fold_1"
    model_name = "BankNifty_Prod_LGBM"

    print(f"Registering model from {model_uri} as {model_name}...")

    try:
        registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
        print(
            f"Successfully registered model: {model_name} (Version: {registered_model.version})"
        )

        # Transition to production
        print(
            "Note: In a real production environment, you would use staging/production aliases."
        )
        client.set_registered_model_alias(
            model_name, "champion", registered_model.version
        )
        print(f"Set model version {registered_model.version} as 'champion'.")

    except Exception as e:
        print(f"Failed to register model: {e}")


if __name__ == "__main__":
    run()
