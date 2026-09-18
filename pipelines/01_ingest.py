from __future__ import annotations

import glob
import os
from pathlib import Path

import polars as pl

from src.common.config import load_yaml
from src.data.ingest import load_source, write_bronze


def discover_sources(config: dict) -> list[str]:
    sources = sorted(glob.glob(config["source"]["raw_glob"]))
    if sources:
        return sources
    for fallback in config["ingestion"].get("allow_legacy_files", []):
        if Path(fallback).exists():
            return [fallback]
    raise FileNotFoundError("No configured raw data source was found")


def run() -> None:
    config = load_yaml(os.getenv("DATA_CONFIG", "configs/data/banknifty.yaml"))
    sources = discover_sources(config)
    frames = [load_source(p, input_timezone=config["source"]["timezone"]) for p in sources]
    df = pl.concat(frames, how="diagonal_relaxed").sort("timestamp")
    if df.get_column("timestamp").n_unique() != df.height:
        raise ValueError("Duplicate timestamps found during ingestion; source data was not silently deduplicated")
    write_bronze(df, config["ingestion"]["bronze_path"])
    print(f"Ingested {len(sources)} source file(s), {df.height} canonical 1m rows")


if __name__ == "__main__":
    run()
