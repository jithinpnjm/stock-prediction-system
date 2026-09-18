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
    FeatureContract("range", "5m", "current bar", "5m close"),
    FeatureContract("body", "5m", "current bar", "5m close"),
    FeatureContract("upper_wick", "5m", "current bar", "5m close"),
    FeatureContract("lower_wick", "5m", "current bar", "5m close"),
    FeatureContract("atr_", "5m", "rolling prior/current bars", "5m close"),
    FeatureContract("return_", "5m", "past bars", "5m close"),
    FeatureContract("realized_vol_", "5m", "past bars", "5m close"),
    FeatureContract("cluster_", "5m", "current causal cluster", "5m close"),
    FeatureContract("opening_range_", "5m", "completed opening window", "5m close"),
    FeatureContract("causal_swing_", "5m", "past bars", "5m close"),
    FeatureContract("support_", "5m", "past bars", "5m close"),
    FeatureContract("resistance_", "5m", "past bars", "5m close"),
    FeatureContract("htf_", "5m", "completed higher timeframe", "5m close"),
    FeatureContract("minute_", "1m", "current completed 5m bucket", "5m close"),
    FeatureContract("cusum_", "5m", "past/current returns", "5m close"),
)


def assert_feature_frame_is_pre_label(df: pl.DataFrame) -> None:
    forbidden = {
        "label",
        "event_end",
        "barrier_type",
        "barrier_time",
        "mfe_long",
        "mae_long",
        "mfe_short",
        "mae_short",
    }
    leaked = sorted(forbidden & set(df.columns))
    if leaked:
        raise ValueError(f"Label/outcome columns present in feature frame: {leaked}")
