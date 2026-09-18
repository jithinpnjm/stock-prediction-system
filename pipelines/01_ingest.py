from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import yaml

from src.data.ingest import read_source
from src.data.schemas import validate_required_schema


def run():
    config = yaml.safe_load(Path("configs/data/default.yaml").read_text()) or {}
    source_semantics = config.get("source_timestamp_semantics", "start")
    source_timezone = config.get("timezone", "Asia/Kolkata")
    source = os.getenv("RAW_SOURCE_PATH", config.get("raw_source", "data/raw/fyers"))

    legacy_candidates = [
        Path("data/banknifty_spot_1m.csv"),
        Path("data/raw/1m_data.parquet"),
    ]
    try:
        df = read_source(
            source,
            naive_timezone=source_timezone,
            source_timestamp_semantics=source_semantics,
        )
    except FileNotFoundError:
        for candidate in legacy_candidates:
            if candidate.exists():
                df = read_source(
                    candidate,
                    naive_timezone=source_timezone,
                    source_timestamp_semantics=source_semantics,
                )
                break
        else:
            raise
    validate_required_schema(df)
    out = Path(os.getenv("BRONZE_PATH", "data/bronze/validated_1m.parquet"))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    print(f"ingested {df.height} rows -> {out}")


if __name__ == "__main__":
    run()
