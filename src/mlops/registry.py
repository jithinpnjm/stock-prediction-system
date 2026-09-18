from __future__ import annotations
import mlflow
from mlflow.tracking import MlflowClient

def register(run_id:str,artifact_path:str,name:str,alias:str|None=None):
    version=mlflow.register_model(f"runs:/{run_id}/{artifact_path}",name)
    if alias: MlflowClient().set_registered_model_alias(name,alias,version.version)
    return version
