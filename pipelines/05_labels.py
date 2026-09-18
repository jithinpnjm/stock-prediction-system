from __future__ import annotations

import os

import polars as pl

from src.common.config import load_yaml
from src.common.contracts import LabelConfig
from src.labels.triple_barrier import add_path_statistics, apply_triple_barrier_labels
from src.labels.timing import add_barrier_timing_features


def run() -> None:
    cfg = load_yaml(os.getenv("LABEL_CONFIG", "configs/labels/baseline.yaml"))
    data_cfg = load_yaml(os.getenv("DATA_CONFIG", "configs/data/banknifty.yaml"))
    features = pl.read_parquet("data/gold/features_v1.parquet")
    one = pl.read_parquet(data_cfg["ingestion"]["bronze_path"])

    label_cfg = LabelConfig(
        target_points=float(cfg["target_points"]),
        stop_points=float(cfg["stop_points"]),
        horizon_bars=int(cfg["horizon_bars"]),
        entry_delay_minutes=int(cfg.get("entry_delay_minutes", 1)),
    )
    labeled = apply_triple_barrier_labels(features, one, label_cfg)
    labeled = add_path_statistics(
        labeled,
        one,
        horizon_bars=label_cfg.horizon_bars,
    )
    labeled = add_barrier_timing_features(labeled)

    os.makedirs("data/gold", exist_ok=True)
    labeled.write_parquet("data/gold/dataset_v1.parquet", compression="zstd")
    print(f"Labeled dataset: {labeled.height} rows")


if __name__ == "__main__":
    run()
