from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
import mlflow.lightgbm
import numpy as np
import polars as pl
import yaml
from sklearn.metrics import log_loss

from src.mlops.lineage import Lineage
from src.mlops.mlflow_utils import log_lineage, log_resolved_config
from src.models.lightgbm import predict_proba, train_lightgbm_classifier
from src.validation.purged_cv import PurgedTimeSeriesSplit


def run():
    dataset=pl.read_parquet("data/ml/training_dataset.parquet").sort("timestamp")
    feature_columns=json.loads(Path("data/ml/feature_schema.json").read_text())["feature_columns"]
    X=dataset.select(feature_columns).to_numpy()
    y=dataset["label"].to_numpy()
    timestamps=dataset["timestamp"].dt.epoch("ns").to_numpy()
    event_end=dataset["event_end_timestamp"].dt.epoch("ns").to_numpy()

    vcfg=yaml.safe_load(Path("configs/validation/default.yaml").read_text())
    mcfg=yaml.safe_load(Path("configs/models/lightgbm.yaml").read_text())
    embargo_ns=int(vcfg["embargo_bars"])*5*60*1_000_000_000
    splitter=PurgedTimeSeriesSplit(int(vcfg["n_splits"]),embargo=embargo_ns)

    rows=[]
    mlflow.set_experiment("BankNifty_Baseline_LGBM")
    with mlflow.start_run(run_name=os.getenv("MLFLOW_RUN_NAME","lgbm_walkforward")):
        lineage=Lineage.create(
            dataset_id="banknifty_5m_v1",
            feature_version="price_action_v1",
            label_version="triple_barrier_200_70_v1",
            validation_version="purged_time_series_v1",
        )
        log_lineage(lineage.to_dict())
        log_resolved_config({"model":mcfg,"validation":vcfg})
        mlflow.log_param("n_rows",len(y))
        mlflow.log_param("n_features",len(feature_columns))
        mlflow.log_param("n_splits",splitter.get_n_splits())

        params=dict(mcfg)
        rounds=int(params.pop("num_boost_round",1000))
        params.pop("early_stopping_rounds",None)

        for fold,(train_idx,val_idx) in enumerate(splitter.split(timestamps,event_end),1):
            if len(train_idx)<int(vcfg["min_train_rows"]):
                continue
            model=train_lightgbm_classifier(
                X[train_idx],y[train_idx],X[val_idx],y[val_idx],
                params=params,num_boost_round=rounds
            )
            p=predict_proba(model,X[val_idx])
            loss=log_loss(y[val_idx],p,labels=[-1,0,1])
            mlflow.log_metric(f"fold_{fold}_log_loss",loss)
            for idx,row in zip(val_idx,p):
                rows.append({
                    "timestamp":dataset["timestamp"][idx],
                    "label":int(y[idx]),
                    "p_short":float(row[0]),
                    "p_none":float(row[1]),
                    "p_long":float(row[2]),
                    "fold":fold,
                })
            mlflow.lightgbm.log_model(model,artifact_path=f"model_fold_{fold}")

    if not rows:
        raise RuntimeError("No valid OOF folds were produced")
    out=pl.DataFrame(rows).sort("timestamp")
    Path("data/predictions").mkdir(parents=True,exist_ok=True)
    out.write_parquet("data/predictions/lgbm_oof.parquet")
    print(f"wrote {out.height} OOF predictions")


if __name__=="__main__":
    run()
