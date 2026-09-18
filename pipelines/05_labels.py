from __future__ import annotations

import yaml
import polars as pl

from src.labels.triple_barrier import apply_triple_barrier_labels


def run():
    cfg=yaml.safe_load(open("configs/labels/triple_barrier.yaml"))
    features=pl.read_parquet("data/silver/5m_features.parquet")
    source=pl.read_parquet("data/bronze/validated_1m.parquet")
    labeled=apply_triple_barrier_labels(
        features,source,
        target_pts=float(cfg["target_points"]),
        stop_pts=float(cfg["stop_points"]),
        max_horizon_minutes=int(cfg["max_horizon_minutes"]),
        direction=cfg["direction"],
    )
    labeled.write_parquet("data/gold/dataset_v1.parquet")
    print(f"labeled dataset: {labeled.shape}")


if __name__=="__main__":
    run()
