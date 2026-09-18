from __future__ import annotations

import subprocess


STAGES=[
    "01_ingest.py","02_validate.py","03_aggregate_5m.py","04_features.py",
    "05_labels.py","06_build_dataset.py","07_train.py","08_validate.py",
    "09_backtest.py","10_report.py",
]


def run_pipeline():
    for stage in STAGES:
        subprocess.run(["python",f"pipelines/{stage}"],check=True)
