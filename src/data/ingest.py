from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import polars as pl

from .schemas import REQUIRED_COLUMNS, coerce_canonical_schema


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_naive_timestamp(df: pl.DataFrame, timezone: str) -> pl.DataFrame:
    dtype = df["timestamp"].dtype
    if isinstance(dtype, pl.Datetime) and dtype.time_zone:
        return df.with_columns(
            pl.col("timestamp").dt.convert_time_zone(timezone).alias("timestamp")
        )
    return df.with_columns(
        pl.col("timestamp")
        .cast(pl.Datetime(time_zone=None))
        .dt.replace_time_zone(timezone)
        .alias("timestamp")
    )


def read_source(path: str | Path, *, naive_timezone: str = "Asia/Kolkata") -> pl.DataFrame:
    """Read one source file, directory, or glob into canonical 1m schema."""
    p = Path(path)
    if any(ch in str(p) for ch in "*?["):
        matches = sorted(Path().glob(str(p)))
        return read_sources(matches, naive_timezone=naive_timezone)
    if p.is_dir():
        return read_sources(
            sorted([*p.glob("*.parquet"), *p.glob("*.csv")]),
            naive_timezone=naive_timezone,
        )
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix.lower() == ".csv":
        df = pl.from_pandas(pd.read_csv(p))
    elif p.suffix.lower() in {".parquet", ".pq"}:
        df = pl.read_parquet(p)
    else:
        raise ValueError(f"Unsupported source format: {p.suffix}")

    rename = {}
    for old in ("datetime", "date", "ts", "epoch"):
        if old in df.columns and "timestamp" not in df.columns:
            rename[old] = "timestamp"
            break
    if rename:
        df = df.rename(rename)
    if "timestamp" not in df.columns:
        raise ValueError(f"{p} has no timestamp column")
    if df["timestamp"].dtype == pl.Utf8:
        parsed = df["timestamp"].str.to_datetime(strict=False)
        if parsed.null_count() > 0:
            raise ValueError(
                f"{p}: {parsed.null_count()} of {len(parsed)} timestamp values failed to parse"
            )
        df = df.with_columns(parsed.alias("timestamp"))
    return _normalize_naive_timestamp(coerce_canonical_schema(df), naive_timezone)


def read_sources(
    paths: Iterable[str | Path],
    *,
    naive_timezone: str = "Asia/Kolkata",
) -> pl.DataFrame:
    frames = [read_source(path, naive_timezone=naive_timezone) for path in paths]
    if not frames:
        raise FileNotFoundError("No source files found")
    return pl.concat(frames).unique(subset=["timestamp"], keep="last").sort("timestamp")


def write_immutable(df: pl.DataFrame, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = sha256_file(destination)
        tmp = destination.with_suffix(destination.suffix + ".new")
        df.write_parquet(tmp)
        new_hash = sha256_file(tmp)
        tmp.unlink()
        if existing != new_hash:
            raise FileExistsError(f"Refusing to overwrite immutable file: {destination}")
        return
    df.write_parquet(destination)


def write_manifest(
    source_path: str | Path,
    output_path: str | Path,
    *,
    rows: int,
    sha256: str,
    git_commit: str | None = None,
    transform_version: str = "ingest_v2",
) -> None:
    payload = {
        "source": str(source_path),
        "output": str(output_path),
        "rows": rows,
        "sha256": sha256,
        "transform_version": transform_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit or os.getenv("GIT_COMMIT", "unknown"),
        "required_columns": list(REQUIRED_COLUMNS),
    }
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
