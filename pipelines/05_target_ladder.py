from __future__ import annotations

from pathlib import Path

import yaml
import polars as pl

from src.labels.target_ladders import build_target_ladder


def run():
    cfg = yaml.safe_load(open("configs/labels/target_ladder.yaml"))
    events = pl.read_parquet("data/silver/5m_features.parquet")
    source = pl.read_parquet("data/bronze/validated_1m.parquet")
    out = build_target_ladder(
        events,
        source,
        targets=tuple(float(x) for x in cfg["targets"]),
        stop_points=float(cfg["stop_points"]),
        max_horizon_minutes=int(cfg["max_horizon_minutes"]),
    )
    dest = Path("data/gold/target_ladder.parquet")
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(dest)
    print(f"target ladder: {out.shape}")


if __name__ == "__main__":
    run()
