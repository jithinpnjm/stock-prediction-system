from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl

from src.data.ingest import read_source


def test_fyers_start_timestamp_becomes_availability_timestamp(tmp_path):
    p=tmp_path/"source.parquet"
    df=pl.DataFrame({
        "timestamp":[datetime(2026,9,15,9,15)],
        "open":[100.0],"high":[101.0],"low":[99.0],"close":[100.0],"volume":[100.0],
    })
    df.write_parquet(p)
    out=read_source(p)
    assert out["source_timestamp"][0]==datetime(2026,9,15,9,15,tzinfo=ZoneInfo("Asia/Kolkata"))
    assert out["timestamp"][0]==datetime(2026,9,15,9,16,tzinfo=ZoneInfo("Asia/Kolkata"))

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.data.calendar import NSECalendar


def test_calendar_distinguishes_source_and_availability_timestamps():
    tz = ZoneInfo("Asia/Kolkata")
    timestamps = [
        datetime(2026, 9, 15, 9, 15, tzinfo=tz),
        datetime(2026, 9, 15, 9, 16, tzinfo=tz),
        datetime(2026, 9, 15, 15, 29, tzinfo=tz),
        datetime(2026, 9, 15, 15, 30, tzinfo=tz),
    ]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0] * 4,
            "high": [101.0] * 4,
            "low": [99.0] * 4,
            "close": [100.0] * 4,
            "volume": [1.0] * 4,
        }
    )
    calendar = NSECalendar()

    source = calendar.filter_source_session(df)
    availability = calendar.filter_session(df)

    assert source["timestamp"].to_list() == timestamps[:3]
    assert availability["timestamp"].to_list() == timestamps[1:]

def test_utc_source_is_converted_to_market_timezone_before_availability_shift(tmp_path):
    import pandas as pd
    from datetime import timezone

    path = tmp_path / "utc_source.parquet"
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 9, 15, 3, 45, tzinfo=timezone.utc)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [100.0],
        }
    )
    # Force a timezone-aware Arrow/Pandas representation through Parquet.
    df = pl.from_pandas(pd.DataFrame(df.to_dict(as_series=False)))
    # Rebuild the timestamp with an explicit UTC dtype after the conversion above.
    df = df.with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", time_zone="UTC"))
    )
    df.write_parquet(path)

    out = read_source(path)
    assert out["source_timestamp"][0] == datetime(
        2026, 9, 15, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata")
    )
    assert out["timestamp"][0] == datetime(
        2026, 9, 15, 9, 16, tzinfo=ZoneInfo("Asia/Kolkata")
    )
