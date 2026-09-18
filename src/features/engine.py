from __future__ import annotations

import polars as pl

from src.features.candles import add_candle_geometry_features
from src.features.clusters import add_consecutive_cluster_features
from src.features.event_sampling import add_cusum_events
from src.features.market_structure import add_causal_market_structure
from src.features.microstructure import add_1m_inside_5m_features
from src.features.multi_timeframe import add_multi_timeframe_features
from src.features.opening_range import add_opening_range_features
from src.features.support_resistance import add_support_resistance_features
from src.features.swings import add_swing_candidates
from src.features.time_features import add_time_features
from src.features.volatility import add_volatility_features
from src.features.volume import add_volume_features
from src.regimes.causal import add_causal_regime_features


def build_point_in_time_features(
    five_minute: pl.DataFrame,
    one_minute: pl.DataFrame | None = None,
) -> pl.DataFrame:
    out = five_minute.sort("timestamp")
    out = add_candle_geometry_features(out)
    out = add_time_features(out)
    out = add_volatility_features(out)
    out = add_volume_features(out)
    out = add_consecutive_cluster_features(out)
    out = add_opening_range_features(out)
    out = add_causal_market_structure(out)
    out = add_swing_candidates(out)
    out = add_support_resistance_features(out)
    out = add_multi_timeframe_features(out)
    out = add_causal_regime_features(out)
    out = add_cusum_events(out)
    if one_minute is not None:
        out = add_1m_inside_5m_features(one_minute, out)
    return out
