from __future__ import annotations

import mlflow


def register_candidate(model_uri: str, model_name: str, *, alias: str | None = None):
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    if alias:
        from mlflow.tracking import MlflowClient

        MlflowClient().set_registered_model_alias(
            model_name, alias, registered.version
        )
    return registered
