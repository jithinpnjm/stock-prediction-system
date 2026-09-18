from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score,log_loss

from src.validation.calibration import brier_score,expected_calibration_error
from src.validation.statistical_tests import bootstrap_mean_ci


def run():
    df=pl.read_parquet("data/predictions/lgbm_oof.parquet").sort("timestamp")
    y=df["label"].to_numpy()
    p=df.select(["p_short","p_none","p_long"]).to_numpy()
    pred=np.asarray([-1,0,1],dtype=np.int8)[np.argmax(p,axis=1)]
    report={
        "observations":int(len(y)),
        "accuracy":float(accuracy_score(y,pred)),
        "log_loss":float(log_loss(y,p,labels=[-1,0,1])),
        "ece_long":expected_calibration_error((y==1).astype(int),p[:,2]),
        "ece_short":expected_calibration_error((y==-1).astype(int),p[:,0]),
        "brier_long":brier_score((y==1).astype(int),p[:,2]),
        "brier_short":brier_score((y==-1).astype(int),p[:,0]),
    }
    pnl_proxy=np.where(pred==1,(y==1).astype(float),np.where(pred==-1,(y==-1).astype(float),0.0))
    report["proxy_mean_accuracy"]=float(pnl_proxy.mean())
    mean,lo,hi=bootstrap_mean_ci(pnl_proxy,n_boot=2000,seed=42)
    report["proxy_mean_ci95"]=[mean,lo,hi]
    fold_report=[]
    for fold,group in df.group_by("fold",maintain_order=True):
        yy=group["label"].to_numpy()
        pp=group.select(["p_short","p_none","p_long"]).to_numpy()
        fold_report.append({
            "fold":int(fold[0]),
            "rows":group.height,
            "accuracy":float(accuracy_score(yy,np.asarray([-1,0,1])[np.argmax(pp,axis=1)])),
            "log_loss":float(log_loss(yy,pp,labels=[-1,0,1])),
        })
    report["folds"]=fold_report
    Path("reports").mkdir(exist_ok=True)
    Path("reports/validation_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    run()
