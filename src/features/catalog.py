from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class FeatureContract:
    name_prefix: str
    source_timeframe: str
    lookback: str
    available_at: str
    future_information: bool = False


FEATURE_CONTRACTS = (
    FeatureContract("f_", "5m", "point-in-time feature", "5m close"),
)


def assert_feature_frame_is_pre_label(df: pl.DataFrame) -> None:
    forbidden = {
        "label",
        "event_end",
        "event_end_timestamp",
        "barrier_type",
        "barrier_time",
        "barrier_timestamp",
        "entry_price",
        "mfe_long",
        "mae_long",
        "mfe_short",
        "mae_short",
        "mfe_points",
        "mae_points",
        "time_to_barrier_seconds",
    }
    leaked = sorted(forbidden & set(df.columns))
    if leaked:
        raise ValueError(
            "Label/outcome columns present in feature frame: "
            + ", ".join(leaked)
        )


def unknown_feature_columns(df: pl.DataFrame) -> list[str]:
    ignored = {
        "session_date",
        "timestamp",
        "source_timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    return sorted(
        column
        for column in df.columns
        if column not in ignored and column.startswith("f_") is False
    )


def assert_known_feature_columns(df: pl.DataFrame) -> None:
    unknown = unknown_feature_columns(df)
    if unknown:
        raise ValueError(
            "Unregistered/non-feature columns detected in feature frame: "
            + ", ".join(unknown)
        )
