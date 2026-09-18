from __future__ import annotations

import os

import polars as pl

from src.common.config import load_yaml
from src.data.aggregate import (
    aggregate_1m_to_5m,
    validate_aggregation,
)
from src.data.calendar import load_session_overrides


def run() -> None:
    config = load_yaml(
        os.getenv(
            "DATA_CONFIG",
            "configs/data/banknifty.yaml",
        )
    )

    df_1m = pl.read_parquet(
        config["ingestion"]["bronze_path"]
    )
    overrides = load_session_overrides(
        config["ingestion"].get(
            "session_overrides_path"
        )
    )

    df_5m = aggregate_1m_to_5m(
        df_1m,
        session_overrides=overrides,
    )
    validate_aggregation(
        df_5m,
        session_overrides=overrides,
    )

    os.makedirs("data/silver", exist_ok=True)
    df_5m.write_parquet(
        "data/silver/5m_canonical.parquet",
        compression="zstd",
    )
    print(
        f"Canonical 5m dataset: {df_5m.height} rows"
    )


if __name__ == "__main__":
    run()
