from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from .schemas import coerce_canonical_schema


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pl.read_csv(path, try_parse_dates=True)
        rename = {"datetime": "timestamp", "date": "timestamp", "ts": "timestamp"}
        for old, new in rename.items():
            if old in df.columns and new not in df.columns:
                df = df.rename({old: new})
        return coerce_canonical_schema(df)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pl.read_parquet(path)
        if "datetime" in df.columns and "timestamp" not in df.columns:
            df = df.rename({"datetime": "timestamp"})
        return coerce_canonical_schema(df)
    raise ValueError(f"Unsupported source format: {path.suffix}")


def write_immutable(df: pl.DataFrame, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = sha256_file(destination)
        candidate = destination.with_suffix(destination.suffix + ".new")
        df.write_parquet(candidate)
        new_hash = sha256_file(candidate)
        candidate.unlink()
        if existing != new_hash:
            raise FileExistsError(
                f"Refusing to overwrite immutable raw file: {destination}"
            )
        return
    df.write_parquet(destination)


def write_manifest(
    source_path: str | Path,
    output_path: str | Path,
    rows: int,
    sha256: str,
    *,
    git_commit: str | None = None,
    transform_version: str = "ingest_v1",
) -> None:
    payload = {
        "source": str(source_path),
        "output": str(output_path),
        "rows": rows,
        "sha256": sha256,
        "transform_version": transform_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2) + "\n")
