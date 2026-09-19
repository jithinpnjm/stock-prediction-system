from __future__ import annotations

from datetime import date

import polars as pl

from src.data.nse_fo_bhavcopy import parse_banknifty_daily


def _synthetic_bhavcopy() -> pl.DataFrame:
    rows = [
        # futures: near-month first
        {
            "INSTRUMENT": "FUTIDX",
            "SYMBOL": "BANKNIFTY",
            "EXPIRY_DT": "25-Jan-2024",
            "STRIKE_PR": 0.0,
            "OPTION_TYP": "XX",
            "OPEN": 48000,
            "HIGH": 48200,
            "LOW": 47900,
            "CLOSE": 48100.0,
            "SETTLE_PR": 48100.0,
            "CONTRACTS": 1000,
            "VAL_INLAKH": 100.0,
            "OPEN_INT": 500000.0,
            "CHG_IN_OI": 10000.0,
        },
        # options at nearest expiry, two strikes each side
        {
            "INSTRUMENT": "OPTIDX",
            "SYMBOL": "BANKNIFTY",
            "EXPIRY_DT": "03-Jan-2024",
            "STRIKE_PR": 48000.0,
            "OPTION_TYP": "CE",
            "OPEN": 100,
            "HIGH": 120,
            "LOW": 90,
            "CLOSE": 100.0,
            "SETTLE_PR": 100.0,
            "CONTRACTS": 500,
            "VAL_INLAKH": 10.0,
            "OPEN_INT": 20000.0,
            "CHG_IN_OI": 1000.0,
        },
        {
            "INSTRUMENT": "OPTIDX",
            "SYMBOL": "BANKNIFTY",
            "EXPIRY_DT": "03-Jan-2024",
            "STRIKE_PR": 48000.0,
            "OPTION_TYP": "PE",
            "OPEN": 90,
            "HIGH": 110,
            "LOW": 80,
            "CLOSE": 90.0,
            "SETTLE_PR": 90.0,
            "CONTRACTS": 300,
            "VAL_INLAKH": 8.0,
            "OPEN_INT": 15000.0,
            "CHG_IN_OI": 500.0,
        },
        {
            "INSTRUMENT": "OPTIDX",
            "SYMBOL": "BANKNIFTY",
            "EXPIRY_DT": "31-Jan-2024",
            "STRIKE_PR": 48500.0,
            "OPTION_TYP": "CE",
            "OPEN": 50,
            "HIGH": 60,
            "LOW": 40,
            "CLOSE": 50.0,
            "SETTLE_PR": 50.0,
            "CONTRACTS": 100,
            "VAL_INLAKH": 2.0,
            "OPEN_INT": 5000.0,
            "CHG_IN_OI": 100.0,
        },
    ]
    return pl.DataFrame(rows)


def test_parse_banknifty_daily_picks_nearest_expiry_and_computes_pcr():
    raw = _synthetic_bhavcopy()
    out = parse_banknifty_daily(raw, date(2024, 1, 2))
    assert out is not None
    assert out["opt_near_expiry"] == "2024-01-03"
    assert out["fut_close"] == 48100.0
    assert out["opt_total_ce_oi"] == 20000.0
    assert out["opt_total_pe_oi"] == 15000.0
    assert abs(out["opt_pcr_oi"] - (15000.0 / 20000.0)) < 1e-6


def test_parse_returns_none_for_missing_symbol():
    raw = _synthetic_bhavcopy().filter(pl.col("SYMBOL") == "BANKNIFTY").head(0)
    assert parse_banknifty_daily(raw, date(2024, 1, 2)) is None
