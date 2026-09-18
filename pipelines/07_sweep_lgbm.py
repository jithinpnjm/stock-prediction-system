from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import optuna
import polars as pl
import yaml
from sklearn.metrics import log_loss

from src.mlops.lineage import Lineage
from src.mlops.mlflow_utils import log_lineage
from src.models.lightgbm import predict_proba, train_lightgbm_classifier
from src.research.experiment_budget import ExperimentBudget
from src.validation.purged_cv import PurgedTimeSeriesSplit


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument("--trials",type=int,default=100)
    args=parser.parse_args()

    budget=ExperimentBudget(
        max_trials=min(max(args.trials,1),1000),
        experiment_family="lgbm_baseline",
    )
    dataset=pl.read_parquet("data/ml/training_dataset.parquet").sort("timestamp")
    feature_names=json.loads(
        Path("data/ml/feature_schema.json").read_text()
    )["feature_columns"]
    X=dataset.select(feature_names).to_numpy()
    y=dataset["label"].to_numpy()
    timestamps=dataset["timestamp"].dt.epoch(time_unit="ns").to_numpy()
    event_end=dataset["event_end_timestamp"].dt.epoch(time_unit="ns").to_numpy()

    val_cfg=yaml.safe_load(Path("configs/validation/default.yaml").read_text())
    splitter=PurgedTimeSeriesSplit(
        int(val_cfg["n_splits"]),
        int(val_cfg["embargo_bars"])*5*60*1_000_000_000,
    )
    folds=list(splitter.split(timestamps,event_end))
    if not folds:
        raise RuntimeError("no validation folds available")

    mlflow.set_experiment("BankNifty_LGBM_Sweeps")

    def objective(trial:optuna.Trial)->float:
        budget.validate_trial(trial.number)
        params={
            "num_leaves":trial.suggest_int("num_leaves",15,127,step=8),
            "learning_rate":trial.suggest_float("learning_rate",0.01,0.1,log=True),
            "min_data_in_leaf":trial.suggest_int("min_data_in_leaf",20,150),
            "feature_fraction":trial.suggest_float("feature_fraction",0.6,1.0),
            "bagging_fraction":trial.suggest_float("bagging_fraction",0.6,1.0),
            "bagging_freq":1,
            "verbosity":-1,
        }
        losses=[]
        with mlflow.start_run(
            run_name=f"optuna_trial_{trial.number}",
            nested=True,
        ):
            for train_idx,val_idx in folds:
                model=train_lightgbm_classifier(
                    X[train_idx],y[train_idx],X[val_idx],y[val_idx],
                    params=params,num_boost_round=700,
                )
                losses.append(
                    log_loss(
                        y[val_idx],
                        predict_proba(model,X[val_idx]),
                        labels=[-1,0,1],
                    )
                )
            value=float(np.mean(losses))
            mlflow.log_params(params)
            mlflow.log_metric("cv_log_loss",value)
            mlflow.log_param("trial_number",trial.number)
            log_lineage(Lineage.create(
                dataset_id="banknifty_5m_v1",
                feature_version="price_action_v2",
                label_version="triple_barrier_200_70_v1",
                validation_version="purged_time_series_v1",
            ).to_dict())
        return value

    with mlflow.start_run(run_name="lgbm_optuna_parent"):
        study=optuna.create_study(direction="minimize")
        study.optimize(objective,n_trials=budget.max_trials)
        mlflow.log_metric("best_cv_log_loss",float(study.best_value))
        mlflow.log_param("trials_completed",len(study.trials))
        print(f"best_value={study.best_value}")
        print(f"best_params={study.best_params}")


if __name__=="__main__":
    run()
