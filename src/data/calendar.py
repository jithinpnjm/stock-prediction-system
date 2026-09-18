from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from src.common.contracts import MARKET_TIMEZONE, SESSION_CLOSE, SESSION_OPEN

IST = ZoneInfo(MARKET_TIMEZONE)


def load_holidays(path: str | Path | None = None) -> set[date]:
    if path is None:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    frame = pl.read_csv(p)
    if "date" not in frame.columns:
        raise ValueError(f"Holiday file {p} must contain a date column")
    return {
        date.fromisoformat(str(value))
        for value in frame.get_column("date").to_list()
        if str(value).strip()
    }


def is_trading_day(value: date, holidays: set[date] | None = None) -> bool:
    return value.weekday() < 5 and value not in (holidays or set())


def session_open(session_date: date) -> datetime:
    return datetime.combine(session_date, SESSION_OPEN, tzinfo=IST)


def session_close(session_date: date) -> datetime:
    return datetime.combine(session_date, SESSION_CLOSE, tzinfo=IST)


def expected_1m_timestamps(session_date: date) -> list[datetime]:
    current = session_open(session_date)
    end = session_close(session_date)
    out: list[datetime] = []
    while current < end:
        out.append(current)
        current += timedelta(minutes=1)
    return out


def expected_5m_close_timestamps(session_date: date) -> list[datetime]:
    current = session_open(session_date) + timedelta(minutes=5)
    end = session_close(session_date)
    out: list[datetime] = []
    while current <= end:
        out.append(current)
        current += timedelta(minutes=5)
    return out
