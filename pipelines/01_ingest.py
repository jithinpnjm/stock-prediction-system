from __future__ import annotations

import os
from pathlib import Path

import polars as pl

from src.data.ingest import read_source
from src.data.schemas import validate_required_schema


def run():
    source=os.getenv("RAW_SOURCE_PATH","data/raw/fyers")
    legacy_candidates=[
        Path("data/banknifty_spot_1m.csv"),
        Path("data/raw/1m_data.parquet"),
    ]
    try:
        df=read_source(source)
    except FileNotFoundError:
        for candidate in legacy_candidates:
            if candidate.exists():
                df=read_source(candidate)
                break
        else:
            raise
    validate_required_schema(df)
    out=Path(os.getenv("BRONZE_PATH","data/bronze/validated_1m.parquet"))
    out.parent.mkdir(parents=True,exist_ok=True)
    df.write_parquet(out)
    print(f"ingested {df.height} rows -> {out}")


if __name__=="__main__":
    run()
