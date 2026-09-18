from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl

from src.common.contracts import MARKET_TIMEZONE, SESSION_CLOSE, SESSION_OPEN


def load_holidays(path: str | Path | None = None) -> set[date]:
    if path is None:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    holidays: set[date] = set()
    for value in pl.read_csv(p).get_column("date").to_list():
        holidays.add(date.fromisoformat(str(value)))
    return holidays


def is_weekday(value: date) -> bool:
    return value.weekday() < 5


def is_trading_day(value: date, holidays: set[date] | None = None) -> bool:
    return is_weekday(value) and value not in (holidays or set())


def expected_1m_timestamps(
    session_date: date,
) -> list[datetime]:
    start = datetime.combine(session_date, SESSION_OPEN).replace(
        tzinfo=__import__("zoneinfo").ZoneInfo(MARKET_TIMEZONE)
    )
    close = datetime.combine(session_date, SESSION_CLOSE).replace(
        tzinfo=__import__("zoneinfo").ZoneInfo(MARKET_TIMEZONE)
    )
    out: list[datetime] = []
    current = start
    while current < close:
        out.append(current)
        current += timedelta(minutes=1)
    return out


def expected_5m_close_timestamps(session_date: date) -> list[datetime]:
    start = datetime.combine(session_date, SESSION_OPEN).replace(
        tzinfo=__import__("zoneinfo").ZoneInfo(MARKET_TIMEZONE)
    )
    out: list[datetime] = []
    current = start + timedelta(minutes=5)
    close = datetime.combine(session_date, SESSION_CLOSE).replace(
        tzinfo=__import__("zoneinfo").ZoneInfo(MARKET_TIMEZONE)
    )
    while current <= close:
        out.append(current)
        current += timedelta(minutes=5)
    return out
