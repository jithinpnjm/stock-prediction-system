import polars as pl

from src.backtest.decision import build_trade_decisions


def test_low_confidence_returns_no_trade():
    df = pl.DataFrame(
        {
            "timestamp": [1],
            "p_short": [0.34],
            "p_flat": [0.33],
            "p_long": [0.33],
        }
    )
    out = build_trade_decisions(df, min_confidence=0.55, min_edge=0.08)
    assert out["signal"][0] == 0
