from __future__ import annotations

import glob
import os
from pathlib import Path

import polars as pl

from src.common.config import load_yaml
from src.data.ingest import load_source, write_bronze


def discover_sources(config: dict) -> list[str]:
    pattern = config["source"]["raw_glob"]
    sources = sorted(glob.glob(pattern))
    if sources:
        return sources
    for fallback in config["ingestion"].get("allow_legacy_files", []):
        if Path(fallback).exists():
            return [fallback]
    raise FileNotFoundError(
        f"No raw sources found for {pattern} or configured legacy paths"
    )


def run() -> None:
    config = load_yaml(os.getenv("DATA_CONFIG", "configs/data/banknifty.yaml"))
    sources = discover_sources(config)
    frames = [load_source(p, input_timezone=config["source"]["timezone"]) for p in sources]

    df = (
        pl.concat(frames, how="diagonal_relaxed")
        .sort("timestamp")
        .unique(subset=["timestamp"], keep="last", maintain_order=True)
    )
    bronze = config["ingestion"]["bronze_path"]
    write_bronze(df, bronze)
    print(f"Ingested {len(sources)} source file(s), {df.height} canonical 1m rows -> {bronze}")


if __name__ == "__main__":
    run()
