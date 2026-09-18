from __future__ import annotations

import yaml
import polars as pl

from src.features.candles import add_candle_geometry_features
from src.features.clusters import add_candle_cluster_features
from src.features.event_sampling import add_event_sampling_features
from src.features.market_structure import add_market_structure_features
from src.features.opening_range import add_opening_range_features
from src.features.session_context import add_session_context_features
from src.features.support_resistance import add_support_resistance_features
from src.features.swings import add_swing_features
from src.features.time_features import add_time_features
from src.features.volatility import add_volatility_features


def run():
    cfg=yaml.safe_load(open("configs/features/default.yaml"))
    df=pl.read_parquet("data/silver/5m_canonical.parquet")
    df=add_candle_geometry_features(df)
    df=add_time_features(df)
    df=add_session_context_features(df)
    df=add_candle_cluster_features(df,int(cfg["cluster_max_bars"]))
    df=add_volatility_features(df,tuple(cfg["atr_periods"]))
    df=add_swing_features(df,int(cfg["swing_lookback"]))
    df=add_support_resistance_features(df,int(cfg["support_resistance_lookback"]))
    df=add_market_structure_features(df)
    df=add_opening_range_features(df,tuple(cfg["opening_range_windows"]))
    df=add_event_sampling_features(df,float(cfg["cusum_threshold_multiple"]))
    df.write_parquet("data/silver/5m_features.parquet")
    print(f"feature dataset: {df.shape}")


if __name__=="__main__":
    run()
