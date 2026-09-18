from __future__ import annotations

import argparse

import mlflow
from mlflow.tracking import MlflowClient


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--artifact-path",default="model_fold_1")
    parser.add_argument("--name",default="BankNifty_LGBM")
    parser.add_argument("--alias",default=None)
    args=parser.parse_args()
    version=mlflow.register_model(
        model_uri=f"runs:/{args.run_id}/{args.artifact_path}",name=args.name
    )
    if args.alias:
        MlflowClient().set_registered_model_alias(args.name,args.alias,version.version)
        print(f"registered version={version.version}, alias={args.alias}")
    else:
        print(f"registered version={version.version}; no alias changed")


if __name__=="__main__":
    run()
