"""Black-Scholes option premium simulation for backtesting an options
BUYER's real P&L (not spot points). We don't have historical intraday
option tick data (NSE's free bhavcopy is EOD-only), so premiums are
simulated from spot price + an IV input (India VIX as a market-wide
proxy, since historical Bank-Nifty-specific IV isn't available) via
Black-Scholes. This is an approximation -- real option prices can
diverge from BS (skew, liquidity, bid/ask) -- but it is far more honest
than treating spot points as option P&L, which ignores theta and IV
entirely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from scipy.stats import norm

RISK_FREE_RATE = 0.07  # approx Indian short-term rate
STRIKE_STEP = 100.0  # Bank Nifty option strikes are in multiples of 100


def atm_strike(spot: float, step: float = STRIKE_STEP) -> float:
    return round(spot / step) * step


def bs_price(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    iv: float,
    option_type: str,
    r: float = RISK_FREE_RATE,
) -> float:
    """Black-Scholes European option premium. Returns intrinsic value
    (floor of 0) if time_to_expiry or iv is non-positive (expired /
    degenerate)."""
    if time_to_expiry_years <= 0 or iv <= 0:
        if option_type == "CE":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    d1 = (math.log(spot / strike) + (r + 0.5 * iv**2) * time_to_expiry_years) / (
        iv * math.sqrt(time_to_expiry_years)
    )
    d2 = d1 - iv * math.sqrt(time_to_expiry_years)

    if option_type == "CE":
        return spot * norm.cdf(d1) - strike * math.exp(-r * time_to_expiry_years) * norm.cdf(d2)
    return strike * math.exp(-r * time_to_expiry_years) * norm.cdf(-d2) - spot * math.exp(
        -r * time_to_expiry_years
    ) * norm.cdf(-d1)


def next_monthly_expiry(d: date) -> date:
    """Approximate Bank Nifty monthly expiry: the last Thursday of the
    contract month (the long-standing NSE convention through most of
    this dataset's history; exact weekday has changed by circular at
    various points, so this is an approximation for backtesting, not a
    precise historical expiry calendar)."""
    if d.month == 12:
        next_month_first = date(d.year + 1, 1, 1)
    else:
        next_month_first = date(d.year, d.month + 1, 1)
    last_day_of_month = next_month_first - _one_day()
    last_thursday = last_day_of_month
    while last_thursday.weekday() != 3:  # Monday=0 ... Thursday=3
        last_thursday = _sub_days(last_thursday, 1)
    if last_thursday < d:
        # d is past this month's expiry -- but since d is always <= last
        # day of its own month, this only triggers same-day-as-expiry
        # edge cases; roll to next month's expiry.
        return next_monthly_expiry(next_month_first)
    return last_thursday


def _one_day():
    from datetime import timedelta

    return timedelta(days=1)


def _sub_days(d: date, n: int) -> date:
    from datetime import timedelta

    return d - timedelta(days=n)


@dataclass(frozen=True)
class OptionTrade:
    entry_date: date
    exit_date: date
    side: str  # "CE" (bullish/long call) or "PE" (bearish/long put)
    strike: float
    entry_premium: float
    exit_premium: float
    entry_iv: float
    exit_iv: float
    pnl_per_lot: float  # per unit; multiply by lot size externally


def simulate_option_buy(
    *,
    entry_date: date,
    exit_date: date,
    entry_spot: float,
    exit_spot: float,
    entry_iv: float,
    exit_iv: float,
    direction: str,  # "long" or "short" (spot view) -> buys CE or PE respectively
) -> OptionTrade:
    option_type = "CE" if direction == "long" else "PE"
    strike = atm_strike(entry_spot)
    expiry = next_monthly_expiry(entry_date)

    entry_tte = max((expiry - entry_date).days, 0) / 365.0
    exit_tte = max((expiry - exit_date).days, 0) / 365.0

    entry_premium = bs_price(entry_spot, strike, entry_tte, entry_iv, option_type)
    exit_premium = bs_price(exit_spot, strike, exit_tte, exit_iv, option_type)

    return OptionTrade(
        entry_date=entry_date,
        exit_date=exit_date,
        side=option_type,
        strike=strike,
        entry_premium=entry_premium,
        exit_premium=exit_premium,
        entry_iv=entry_iv,
        exit_iv=exit_iv,
        pnl_per_lot=exit_premium - entry_premium,
    )
