from __future__ import annotations

from pathlib import Path
import polars as pl
import yaml

from src.labels.triple_barrier import apply_triple_barrier_labels


def run():
    cfg = yaml.safe_load(Path("configs/labels/triple_barrier.yaml").read_text())
    features = pl.read_parquet("data/silver/5m_features.parquet")
    source = pl.read_parquet("data/bronze/validated_1m.parquet")
    minute_of_day = pl.col("timestamp").dt.hour().cast(pl.Int32) * 60 + pl.col(
        "timestamp"
    ).dt.minute().cast(pl.Int32)
    features = features.filter((minute_of_day >= 9 * 60 + 30) & (minute_of_day <= 15 * 60))
    labeled = apply_triple_barrier_labels(
        features,
        source,
        target_pts=float(cfg["target_points"]),
        stop_pts=float(cfg["stop_points"]),
        max_horizon_minutes=int(cfg["max_horizon_minutes"]),
        direction=cfg["direction"],
    )
    labeled.write_parquet("data/gold/dataset_v1.parquet")
    print(f"labeled dataset: {labeled.shape}")


if __name__ == "__main__":
    run()
