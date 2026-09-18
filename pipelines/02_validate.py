from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl

from src.common.config import load_yaml
from src.data.validate import validate_1m


def run() -> None:
    config = load_yaml(os.getenv("DATA_CONFIG", "configs/data/banknifty.yaml"))
    path = config["ingestion"]["bronze_path"]
    df = pl.read_parquet(path)
    report = validate_1m(
        df,
        holidays_path=config["ingestion"].get("holidays_path"),
        require_complete_sessions=os.getenv("ALLOW_INCOMPLETE_SESSIONS", "0") != "1",
    )

    Path("artifacts/validation").mkdir(parents=True, exist_ok=True)
    Path("artifacts/validation/1m_report.json").write_text(
        json.dumps(report.__dict__, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(report)
    if not report.ok:
        raise SystemExit("Canonical 1m data quality gates failed")


if __name__ == "__main__":
    run()
