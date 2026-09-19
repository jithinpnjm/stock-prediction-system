from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

from src.features.candles import add_candle_geometry_features
from src.features.clusters import add_candle_cluster_features
from src.features.compression_zone import add_compression_zone_features
from src.features.event_sampling import add_event_sampling_features
from src.features.historical_intraday import add_historical_intraday_context
from src.features.market_structure import add_market_structure_features
from src.features.microstructure import add_1m_inside_5m_features
from src.features.multitimeframe import add_multi_timeframe_features
from src.features.opening_range import add_opening_range_features
from src.features.session_context import add_session_context_features
from src.features.support_resistance import add_support_resistance_features
from src.features.swings import add_swing_features
from src.features.time_features import add_time_features
from src.features.volatility import add_volatility_features


def run():
    cfg = yaml.safe_load(Path("configs/features/default.yaml").read_text())
    df5 = pl.read_parquet("data/silver/5m_canonical.parquet")
    source = pl.read_parquet("data/bronze/validated_1m.parquet")

    df = add_candle_geometry_features(df5)
    df = add_time_features(df)
    df = add_session_context_features(df)
    df = add_candle_cluster_features(df, int(cfg["cluster_max_bars"]))
    df = add_volatility_features(df, tuple(cfg["atr_periods"]))
    df = add_multi_timeframe_features(df, tuple(cfg["multi_timeframes"]))
    if bool(cfg.get("enable_historical_intraday", True)):
        df = add_historical_intraday_context(
            df, int(cfg.get("historical_intraday_lookback_days", 20))
        )
    df = add_swing_features(df, int(cfg["swing_lookback"]))
    df = add_support_resistance_features(df, int(cfg["support_resistance_lookback"]))
    df = add_market_structure_features(df)
    df = add_opening_range_features(df, tuple(cfg["opening_range_windows"]))
    df = add_1m_inside_5m_features(source, df)
    df = add_event_sampling_features(df, float(cfg["cusum_threshold_multiple"]))
    df = add_compression_zone_features(df)

    if bool(cfg.get("enable_vwap", False)):
        from src.features.session_vwap import add_session_vwap

        df = add_session_vwap(df)

    if bool(cfg.get("enable_fractional_diff", False)):
        from src.features.fractional_diff import add_fractional_diff_feature

        df = add_fractional_diff_feature(
            df,
            d=float(cfg.get("fractional_diff_d", 0.4)),
            threshold=float(cfg.get("fractional_diff_threshold", 1e-5)),
            max_lags=int(cfg.get("fractional_diff_max_lags", 200)),
        )

    dest = Path("data/silver/5m_features.parquet")
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dest)
    print(f"feature dataset: {df.shape}")


if __name__ == "__main__":
    run()
