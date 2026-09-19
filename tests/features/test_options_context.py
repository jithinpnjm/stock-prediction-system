from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import polars as pl

from src.features.options_context import add_options_context_features

_TZ = ZoneInfo("Asia/Kolkata")


def test_options_context_uses_prior_day_snapshot_only(tmp_path):
    bhav = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "fut_close": [48000.0, 48100.0, 48200.0],
            "fut_oi": [500000.0, 510000.0, 520000.0],
            "fut_oi_chg": [10000.0, 10000.0, 10000.0],
            "opt_pcr_oi": [0.6, 0.7, 0.8],
            "opt_pcr_volume": [0.5, 0.6, 0.7],
            "opt_total_ce_oi_chg": [1000.0, 1000.0, 1000.0],
            "opt_total_pe_oi_chg": [500.0, 500.0, 500.0],
            "opt_max_pain_strike": [48000.0, 48000.0, 48200.0],
            "opt_atm_ce_oi": [20000.0, 20000.0, 20000.0],
            "opt_atm_pe_oi": [15000.0, 16000.0, 17000.0],
        }
    )
    bhav_path = tmp_path / "bhav.parquet"
    bhav.write_parquet(bhav_path)

    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 2, 9, 20, tzinfo=_TZ),
                datetime(2024, 1, 3, 9, 20, tzinfo=_TZ),
            ],
            "close": [48050.0, 48150.0],
        }
    )
    out = add_options_context_features(df, bhavcopy_path=bhav_path)
    # Jan 2 should see Jan 1's PCR (0.6), not Jan 2's own (0.7)
    assert out["f_prev_opt_pcr_oi"][0] == 0.6
    # Jan 3 should see Jan 2's PCR (0.7), not Jan 3's own (0.8)
    assert out["f_prev_opt_pcr_oi"][1] == 0.7


def test_options_context_noop_when_file_missing():
    df = pl.DataFrame({"timestamp": [datetime(2024, 1, 2, 9, 20, tzinfo=_TZ)], "close": [48050.0]})
    out = add_options_context_features(df, bhavcopy_path="does/not/exist.parquet")
    assert out.equals(df)
