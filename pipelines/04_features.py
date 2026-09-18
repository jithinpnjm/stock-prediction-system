from __future__ import annotations

import os

import polars as pl

from src.common.config import load_yaml
from src.features.engine import build_point_in_time_features


def run() -> None:
    data_cfg = load_yaml(os.getenv("DATA_CONFIG", "configs/data/banknifty.yaml"))
    five = pl.read_parquet("data/silver/5m_canonical.parquet")
    one = pl.read_parquet(data_cfg["ingestion"]["bronze_path"])
    df = build_point_in_time_features(five, one)
    os.makedirs("data/gold", exist_ok=True)
    df.write_parquet("data/gold/features_v1.parquet", compression="zstd")
    print(f"Feature dataset: {df.height} rows x {df.width} columns")


if __name__ == "__main__":
    run()
