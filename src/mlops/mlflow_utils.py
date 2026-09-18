from __future__ import annotations
import mlflow

def log_lineage(lineage:dict):
    mlflow.log_dict(lineage,"lineage.json")
    mlflow.set_tags({k:str(v) for k,v in lineage.items() if k in {
        "git_commit","dataset_id","feature_version","label_version","validation_version"}})

def log_resolved_config(config:dict):
    mlflow.log_dict(config,"resolved_config.json")
