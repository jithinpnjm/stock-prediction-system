from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml

from src.models.sequence_data import build_sequences
from src.models.sequence_training import sequence_predict_proba,train_sequence_classifier
from src.models.tcn import build_tcn
from src.models.transformer import build_transformer
from src.validation.purged_cv import PurgedTimeSeriesSplit


def run(model_family:str="tcn"):
    cfg=yaml.safe_load(Path("configs/models/sequence.yaml").read_text())
    dataset=pl.read_parquet("data/ml/training_dataset.parquet").sort("timestamp")
    names=json.loads(Path("data/ml/feature_schema.json").read_text())["feature_columns"]
    batch=build_sequences(
        dataset.select(names).to_numpy(),dataset["label"].to_numpy(),
        dataset["timestamp"].dt.epoch(time_unit="ns").to_numpy(),
        dataset["event_end_timestamp"].dt.epoch(time_unit="ns").to_numpy(),
        sequence_length=int(cfg["sequence_length"]),
    )
    embargo_ns=int(cfg.get("embargo_bars",75))*5*60*1_000_000_000
    split=PurgedTimeSeriesSplit(n_splits=2,embargo=embargo_ns)
    folds=list(split.split(batch.timestamps,batch.event_end))
    if not folds: raise RuntimeError("no valid sequence validation split")
    train_idx,val_idx=folds[-1]
    if len(train_idx)<int(cfg.get("min_train_rows",500)):
        raise ValueError("sequence train fold is too small")

    if model_family=="tcn":
        model=build_tcn(batch.X.shape[2])
        train_X=batch.X.transpose(0,2,1)
        result=train_sequence_classifier(
            model,train_X[train_idx],batch.y[train_idx],
            train_X[val_idx],batch.y[val_idx],
            epochs=int(cfg["epochs"]),batch_size=int(cfg["batch_size"]),
            learning_rate=float(cfg["learning_rate"]),patience=int(cfg["patience"]),
            device=None if cfg.get("device","auto")=="auto" else cfg["device"]
        )
        probs=sequence_predict_proba(result.model,train_X[val_idx])
    elif model_family=="transformer":
        model=build_transformer(batch.X.shape[2])
        result=train_sequence_classifier(
            model,batch.X[train_idx],batch.y[train_idx],
            batch.X[val_idx],batch.y[val_idx],
            epochs=int(cfg["epochs"]),batch_size=int(cfg["batch_size"]),
            learning_rate=float(cfg["learning_rate"]),patience=int(cfg["patience"]),
            device=None if cfg.get("device","auto")=="auto" else cfg["device"]
        )
        probs=sequence_predict_proba(result.model,batch.X[val_idx])
    else:
        raise ValueError("model_family must be tcn or transformer")

    print(
        f"{model_family}: epoch={result.best_epoch} "
        f"train_loss={result.train_loss:.5f} val_loss={result.val_loss:.5f} "
        f"device={'cuda' if torch.cuda.is_available() else 'cpu'}"
    )
    out=pl.DataFrame({
        "timestamp":pl.from_numpy(batch.timestamps[val_idx]).cast(
            pl.Datetime("ns",time_zone="Asia/Kolkata")
        ),
        "event_end_timestamp":pl.from_numpy(batch.event_end[val_idx]).cast(
            pl.Datetime("ns",time_zone="Asia/Kolkata")
        ),
        "p_short":probs[:,0],"p_none":probs[:,1],"p_long":probs[:,2],
        "label":batch.y[val_idx],
    })
    Path("data/predictions").mkdir(parents=True,exist_ok=True)
    out.write_parquet(f"data/predictions/{model_family}_oof.parquet")


if __name__=="__main__":
    import sys
    run(sys.argv[1] if len(sys.argv)>1 else "tcn")
