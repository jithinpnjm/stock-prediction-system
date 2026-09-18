from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def runtime_metadata() -> dict[str, str]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "hostname": platform.node(),
    }


def build_lineage(
    *,
    dvc_revision: str = "unknown",
    feature_version: str = "unknown",
    label_version: str = "unknown",
) -> dict[str, str]:
    return {
        **runtime_metadata(),
        "dvc_revision": dvc_revision,
        "feature_version": feature_version,
        "label_version": label_version,
        "experiment_id": os.environ.get("EXPERIMENT_ID", "local"),
    }
