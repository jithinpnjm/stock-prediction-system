from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

from src.features.engine import build_point_in_time_features


def run():
    config = yaml.safe_load(Path("configs/features/default.yaml").read_text()) or {}
    five_minute = pl.read_parquet("data/silver/5m_canonical.parquet")
    one_minute = pl.read_parquet("data/bronze/validated_1m.parquet")

    features = build_point_in_time_features(
        five_minute,
        one_minute,
        config=config,
    )

    output = Path("data/silver/5m_features.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(output)
    print(f"feature dataset: {features.shape}")


if __name__ == "__main__":
    run()
