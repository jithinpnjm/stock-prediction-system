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
    FeatureContract("direction", "5m", "current bar", "5m close"),
    FeatureContract("close_location", "5m", "current bar", "5m close"),
    FeatureContract("range_bps", "5m", "current bar", "5m close"),
    FeatureContract("body_bps", "5m", "current bar", "5m close"),
    FeatureContract("body_pct_range", "5m", "current bar", "5m close"),
    FeatureContract("upper_wick_pct_range", "5m", "current bar", "5m close"),
    FeatureContract("lower_wick_pct_range", "5m", "current bar", "5m close"),
    FeatureContract("atr_", "5m", "rolling current bars", "5m close"),
    FeatureContract("range_to_atr_", "5m", "rolling/current", "5m close"),
    FeatureContract("return_", "5m", "past bars", "5m close"),
    FeatureContract("realized_vol_", "5m", "past bars", "5m close"),
    FeatureContract("rolling_range_", "5m", "past bars", "5m close"),
    FeatureContract("consecutive_length", "5m", "current causal cluster", "5m close"),
    FeatureContract("cluster_", "5m", "current causal cluster", "5m close"),
    FeatureContract("has_cluster_", "5m", "current causal cluster", "5m close"),
    FeatureContract("minutes_", "5m", "clock", "5m close"),
    FeatureContract("time_", "5m", "clock", "5m close"),
    FeatureContract("weekday_", "5m", "calendar", "5m close"),
    FeatureContract("is_opening_", "5m", "clock", "5m close"),
    FeatureContract("is_closing_", "5m", "clock", "5m close"),
    FeatureContract("opening_range_", "5m", "opening window", "5m close"),
    FeatureContract("position_in_opening_", "5m", "opening window", "5m close"),
    FeatureContract("breaks_opening_", "5m", "current bar", "5m close"),
    FeatureContract("prior_", "5m", "past bars", "5m close"),
    FeatureContract("breaks_prior_", "5m", "current bar", "5m close"),
    FeatureContract("distance_to_prior_", "5m", "past structure", "5m close"),
    FeatureContract("causal_swing_", "5m", "past bars", "5m close"),
    FeatureContract("distance_resistance_", "5m", "past bars", "5m close"),
    FeatureContract("distance_support_", "5m", "past bars", "5m close"),
    FeatureContract("resistance_", "5m", "past bars", "5m close"),
    FeatureContract("support_", "5m", "past bars", "5m close"),
    FeatureContract("volume_", "5m", "past/current bars", "5m close"),
    FeatureContract("session_cumulative_", "5m", "session-to-date", "5m close"),
    FeatureContract("session_vwap", "5m", "session-to-date", "5m close"),
    FeatureContract("distance_vwap_", "5m", "session-to-date", "5m close"),
    FeatureContract("regime_", "5m", "causal state", "5m close"),
    FeatureContract("htf_", "5m", "completed higher timeframe", "5m close"),
    FeatureContract("minute_", "1m", "current completed 5m bucket", "5m close"),
    FeatureContract("path_", "1m", "current completed 5m bucket", "5m close"),
    FeatureContract("intrabar_", "1m", "current completed 5m bucket", "5m close"),
    FeatureContract("up_", "1m", "current completed 5m bucket", "5m close"),
    FeatureContract("down_", "1m", "current completed 5m bucket", "5m close"),
    FeatureContract("first_last_", "1m", "current completed 5m bucket", "5m close"),
    FeatureContract("cusum_", "5m", "past/current returns", "5m close"),
)


def assert_feature_frame_is_pre_label(df: pl.DataFrame) -> None:
    forbidden = {"label", "event_end", "barrier_type", "barrier_time", "entry_price", "mfe_long", "mae_long", "mfe_short", "mae_short"}
    leaked = sorted(forbidden & set(df.columns))
    if leaked:
        raise ValueError(f"Label/outcome columns present in feature frame: {leaked}")


def unknown_feature_columns(df: pl.DataFrame) -> list[str]:
    ignored = {"session_date", "timestamp", "open", "high", "low", "close", "volume", "source_1m_count"}
    return sorted(
        column for column in df.columns
        if column not in ignored and not any(column.startswith(c.name_prefix) for c in FEATURE_CONTRACTS)
    )


def assert_known_feature_columns(df: pl.DataFrame) -> None:
    unknown = unknown_feature_columns(df)
    if unknown:
        raise ValueError("Unregistered feature columns detected: " + ", ".join(unknown))
