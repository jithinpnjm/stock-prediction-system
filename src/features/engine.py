from __future__ import annotations

from typing import Any

import polars as pl

from src.features.candles import add_candle_geometry_features
from src.features.clusters import add_candle_cluster_features
from src.features.event_sampling import add_event_sampling_features
from src.features.fractional_diff import add_fractional_diff_feature
from src.features.historical_intraday import add_historical_intraday_context
from src.features.market_structure import add_market_structure_features
from src.features.microstructure import add_1m_inside_5m_features
from src.features.multitimeframe import add_multi_timeframe_features
from src.features.opening_range import add_opening_range_features
from src.features.session_context import add_session_context_features
from src.features.session_vwap import add_session_vwap
from src.features.support_resistance import add_support_resistance_features
from src.features.swings import add_swing_features
from src.features.time_features import add_time_features
from src.features.volatility import add_volatility_features


def build_point_in_time_features(
    five_minute: pl.DataFrame,
    one_minute: pl.DataFrame | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Build the canonical causal feature frame used by the research pipeline."""
    cfg = config or {}
    out = add_candle_geometry_features(five_minute)
    out = add_time_features(out)
    out = add_session_context_features(out)
    out = add_candle_cluster_features(
        out,
        int(cfg.get("cluster_max_bars", 10)),
    )
    out = add_volatility_features(
        out,
        tuple(int(x) for x in cfg.get("atr_periods", (6, 14, 30))),
    )
    out = add_multi_timeframe_features(
        out,
        tuple(int(x) for x in cfg.get("multi_timeframes", (15, 30, 60))),
    )

    if bool(cfg.get("enable_historical_intraday", True)):
        out = add_historical_intraday_context(
            out,
            int(cfg.get("historical_intraday_lookback_days", 20)),
        )

    out = add_swing_features(
        out,
        int(cfg.get("swing_lookback", 3)),
    )
    out = add_support_resistance_features(
        out,
        int(cfg.get("support_resistance_lookback", 48)),
    )
    out = add_market_structure_features(out)
    out = add_opening_range_features(
        out,
        tuple(int(x) for x in cfg.get("opening_range_windows", (3, 6))),
    )

    if one_minute is not None:
        out = add_1m_inside_5m_features(one_minute, out)

    out = add_event_sampling_features(
        out,
        float(cfg.get("cusum_threshold_multiple", 2.0)),
    )

    if bool(cfg.get("enable_vwap", False)):
        out = add_session_vwap(out)

    if bool(cfg.get("enable_fractional_diff", False)):
        # Fractional differencing is inherently sequential; compute it
        # independently per NSE session so state cannot cross the boundary.
        parts = []
        for part in out.partition_by(out["timestamp"].dt.date(), maintain_order=True):
            parts.append(
                add_fractional_diff_feature(
                    part,
                    d=float(cfg.get("fractional_diff_d", 0.4)),
                    threshold=float(cfg.get("fractional_diff_threshold", 1e-5)),
                    max_lags=int(cfg.get("fractional_diff_max_lags", 200)),
                )
            )
        out = pl.concat(parts, how="vertical") if parts else out

    return out
