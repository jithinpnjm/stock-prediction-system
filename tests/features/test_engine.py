from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.features.engine import build_point_in_time_features


def _synthetic_5m(rows_per_day: int = 24) -> pl.DataFrame:
    tz = ZoneInfo("Asia/Kolkata")
    rows = []
    for day_offset in range(2):
        start = datetime(2026, 9, 15 + day_offset, 9, 20, tzinfo=tz)
        for i in range(rows_per_day):
            close = 1000.0 + day_offset * 25.0 + i * 2.0
            rows.append(
                {
                    "timestamp": start + timedelta(minutes=5 * i),
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0 + i,
                }
            )
    return pl.DataFrame(rows)


def test_canonical_feature_engine_is_point_in_time():
    df = _synthetic_5m()
    cfg = {
        "cluster_max_bars": 5,
        "atr_periods": (3, 6, 12),
        "multi_timeframes": (15, 30, 60),
        "swing_lookback": 2,
        "support_resistance_lookback": 8,
        "opening_range_windows": (3, 6),
        "historical_intraday_lookback_days": 20,
        "cusum_threshold_multiple": 2.0,
        "enable_historical_intraday": True,
        "enable_vwap": False,
        "enable_fractional_diff": False,
    }

    prefix = df.head(24)
    suffix = df.tail(24)
    future = suffix.tail(1).with_columns(
        pl.col("timestamp").dt.offset_by("5m").alias("timestamp")
    )
    base = build_point_in_time_features(prefix, config=cfg)
    extended = build_point_in_time_features(
        pl.concat([prefix, future]),
        config=cfg,
    )

    feature_columns = [c for c in base.columns if c.startswith("f_")]
    assert feature_columns
    assert base.select(feature_columns).equals(
        extended.head(base.height).select(feature_columns)
    )


def test_feature_frame_contains_no_label_columns():
    from src.features.catalog import assert_feature_frame_is_pre_label

    df = build_point_in_time_features(
        _synthetic_5m(8),
        config={
            "cluster_max_bars": 3,
            "atr_periods": (2, 3, 6),
            "multi_timeframes": (15, 30, 60),
            "swing_lookback": 1,
            "support_resistance_lookback": 4,
            "opening_range_windows": (2, 3),
            "enable_historical_intraday": False,
            "enable_vwap": False,
            "enable_fractional_diff": False,
        },
    )
    assert_feature_frame_is_pre_label(df)
