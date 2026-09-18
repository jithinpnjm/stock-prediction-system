from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl
import yaml


def load_excluded_dates(
    config_path: str | Path = "configs/data/excluded_sessions.yaml",
) -> set[date]:
    """Dates dropped by apply_known_exclusions. Data-quality validation
    must treat these the same as holidays (i.e. not expect data on
    them), or every re-ingestion will flag them as missing sessions."""
    payload = yaml.safe_load(Path(config_path).read_text()) or {}
    return {date.fromisoformat(item["date"]) for item in payload.get("excluded_dates", [])}


def apply_known_exclusions(
    df: pl.DataFrame,
    config_path: str | Path = "configs/data/excluded_sessions.yaml",
) -> pl.DataFrame:
    """Drop documented bad days/rows (vendor gaps, special abbreviated
    sessions) from a canonical 1m frame. Every exclusion here must have
    a written reason in the config; this never silently drops data."""
    payload = yaml.safe_load(Path(config_path).read_text()) or {}
    out = df

    excluded_dates = load_excluded_dates(config_path)
    if excluded_dates:
        out = out.filter(~pl.col("source_timestamp").dt.date().is_in(excluded_dates))

    excluded_ts = {
        datetime.fromisoformat(item["source_timestamp"])
        for item in payload.get("excluded_timestamps", [])
    }
    if excluded_ts:
        out = out.filter(~pl.col("source_timestamp").dt.replace_time_zone(None).is_in(excluded_ts))
    return out
