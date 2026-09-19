from __future__ import annotations

from datetime import date

from src.backtest.options_pricing import (
    atm_strike,
    bs_price,
    next_monthly_expiry,
    simulate_option_buy,
)


def test_atm_strike_rounds_to_nearest_100():
    assert atm_strike(48123) == 48100.0
    assert atm_strike(48160) == 48200.0


def test_bs_price_call_increases_with_spot():
    low = bs_price(47000, 48000, 15 / 365, 0.15, "CE")
    high = bs_price(49000, 48000, 15 / 365, 0.15, "CE")
    assert high > low


def test_bs_price_put_increases_when_spot_falls():
    low_spot_put = bs_price(47000, 48000, 15 / 365, 0.15, "PE")
    high_spot_put = bs_price(49000, 48000, 15 / 365, 0.15, "PE")
    assert low_spot_put > high_spot_put


def test_bs_price_at_expiry_equals_intrinsic():
    assert bs_price(48500, 48000, 0.0, 0.15, "CE") == 500.0
    assert bs_price(47500, 48000, 0.0, 0.15, "PE") == 500.0


def test_next_monthly_expiry_is_a_thursday_in_the_right_month():
    expiry = next_monthly_expiry(date(2024, 1, 2))
    assert expiry.month == 1
    assert expiry.weekday() == 3


def test_simulate_option_buy_long_call_profits_on_up_move():
    trade = simulate_option_buy(
        entry_date=date(2024, 1, 2),
        exit_date=date(2024, 1, 2),
        entry_spot=48000,
        exit_spot=48500,
        entry_iv=0.14,
        exit_iv=0.14,
        direction="long",
    )
    assert trade.side == "CE"
    assert trade.pnl_per_lot > 0


def test_simulate_option_buy_theta_decay_hurts_flat_day():
    trade = simulate_option_buy(
        entry_date=date(2024, 1, 2),
        exit_date=date(2024, 1, 2),
        entry_spot=48000,
        exit_spot=48000,  # no move at all
        entry_iv=0.14,
        exit_iv=0.14,
        direction="long",
    )
    # same-day theta decay is tiny but should not be positive with zero move
    assert trade.pnl_per_lot <= 0
