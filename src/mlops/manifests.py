from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_manifest(
    path: str | Path,
    *,
    schema: dict,
    row_count: int,
    source: str,
    version: str,
) -> dict:
    p = Path(path)
    return {
        "path": str(p),
        "sha256": sha256_file(p),
        "size_bytes": p.stat().st_size,
        "schema": schema,
        "row_count": row_count,
        "source": source,
        "version": version,
    }


def write_manifest(manifest: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
