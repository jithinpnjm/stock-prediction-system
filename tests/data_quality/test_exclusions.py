from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import yaml

from src.data.exclusions import apply_known_exclusions


def _df():
    tz = ZoneInfo("Asia/Kolkata")
    rows = [
        (datetime(2022, 3, 7, 10, 56, tzinfo=tz), 100.0),
        (datetime(2022, 3, 8, 9, 15, tzinfo=tz), 101.0),
        (datetime(2023, 11, 8, 9, 15, tzinfo=tz), 102.0),
        (datetime(2023, 11, 8, 9, 16, tzinfo=tz), 103.0),
    ]
    return pl.DataFrame(
        {
            "source_timestamp": [r[0] for r in rows],
            "close": [r[1] for r in rows],
        }
    )


def test_excludes_configured_date_and_timestamp(tmp_path: Path):
    cfg = tmp_path / "excluded_sessions.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "excluded_dates": [{"date": "2022-03-07", "reason": "test"}],
                "excluded_timestamps": [
                    {"source_timestamp": "2023-11-08 09:15:00", "reason": "test"}
                ],
            }
        )
    )
    out = apply_known_exclusions(_df(), config_path=cfg)
    remaining = out["source_timestamp"].to_list()
    assert datetime(2022, 3, 7, 10, 56, tzinfo=ZoneInfo("Asia/Kolkata")) not in remaining
    assert datetime(2023, 11, 8, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata")) not in remaining
    assert out.height == 2
