from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.models.sequence_data import build_sequences
from src.models.sequence_training import sequence_predict_proba,train_sequence_classifier
from src.models.tcn import build_tcn
from src.models.transformer import build_transformer


def run(model_family:str="tcn",sequence_length:int=48):
    dataset=pl.read_parquet("data/ml/training_dataset.parquet").sort("timestamp")
    names=json.loads(Path("data/ml/feature_schema.json").read_text())["feature_columns"]
    batch=build_sequences(
        dataset.select(names).to_numpy(),
        dataset["label"].to_numpy(),
        dataset["timestamp"].dt.epoch("ns").to_numpy(),
        sequence_length=sequence_length,
    )
    cut=int(len(batch.y)*0.8)
    if cut<1 or cut>=len(batch.y):
        raise ValueError("sequence dataset is too small")
    if model_family=="tcn":
        model=build_tcn(batch.X.shape[2])
        # TCN expects [batch, features, time]
        result=train_sequence_classifier(
            model,batch.X[:cut].transpose(0,2,1),batch.y[:cut],
            batch.X[cut:].transpose(0,2,1),batch.y[cut:]
        )
        probs=sequence_predict_proba(
            result.model,batch.X[cut:].transpose(0,2,1)
        )
    elif model_family=="transformer":
        model=build_transformer(batch.X.shape[2])
        result=train_sequence_classifier(
            model,batch.X[:cut],batch.y[:cut],
            batch.X[cut:],batch.y[cut:]
        )
        probs=sequence_predict_proba(result.model,batch.X[cut:])
    else:
        raise ValueError("model_family must be tcn or transformer")
    print(
        f"{model_family}: epoch={result.best_epoch} "
        f"train_loss={result.train_loss:.5f} val_loss={result.val_loss:.5f} "
        f"device={'cuda' if torch.cuda.is_available() else 'cpu'}"
    )
    out=pl.DataFrame({
        "timestamp":pl.from_numpy(batch.timestamps[cut:]).cast(pl.Datetime("ns",time_zone="Asia/Kolkata")),
        "p_short":probs[:,0],"p_none":probs[:,1],"p_long":probs[:,2],
        "label":batch.y[cut:],
    })
    Path("data/predictions").mkdir(parents=True,exist_ok=True)
    out.write_parquet(f"data/predictions/{model_family}_holdout.parquet")


if __name__=="__main__":
    import sys
    run(sys.argv[1] if len(sys.argv)>1 else "tcn")
