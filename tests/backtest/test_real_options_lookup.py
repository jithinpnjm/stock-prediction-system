from __future__ import annotations

from datetime import date

from src.backtest.real_options_lookup import build_symbol


def test_build_symbol_format():
    assert build_symbol(date(2026, 4, 1), 55000, "PE") == "NSE:BANKNIFTY26APR55000PE"
    assert build_symbol(date(2026, 12, 15), 50200, "CE") == "NSE:BANKNIFTY26DEC50200CE"
