from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from src.common.contracts import (
    EXPECTED_1M_BARS,
    MARKET_TIMEZONE,
    SESSION_CLOSE,
    SESSION_OPEN,
)

IST = ZoneInfo(MARKET_TIMEZONE)


@dataclass(frozen=True)
class SessionSpec:
    session_open: time
    session_close: time
    expected_1m_bars: int


STANDARD_SESSION = SessionSpec(
    session_open=SESSION_OPEN,
    session_close=SESSION_CLOSE,
    expected_1m_bars=EXPECTED_1M_BARS,
)


def load_holidays(
    path: str | Path | None = None,
) -> set[date]:
    if path is None:
        return set()

    holiday_path = Path(path)
    if not holiday_path.exists():
        return set()

    frame = pl.read_csv(holiday_path)
    if "date" not in frame.columns:
        raise ValueError(
            f"Holiday file {holiday_path} must contain a date column"
        )

    return {
        date.fromisoformat(str(value))
        for value in frame.get_column("date").to_list()
        if str(value).strip()
    }


def load_closed_days(
    path: str | Path | None = None,
) -> set[date]:
    if path is None:
        return set()

    closed_path = Path(path)
    if not closed_path.exists():
        return set()

    if closed_path.suffix.lower() == ".json":
        payload = json.loads(
            closed_path.read_text(encoding="utf-8")
        )
        if not isinstance(payload, list):
            raise ValueError(
                f"Closed-day JSON {closed_path} must contain a list"
            )
        return {
            date.fromisoformat(str(value))
            for value in payload
        }

    return load_holidays(closed_path)


def load_session_overrides(
    path: str | Path | None = None,
) -> dict[date, SessionSpec]:
    if path is None:
        return {}

    override_path = Path(path)
    if not override_path.exists():
        return {}

    frame = pl.read_csv(override_path)
    required = {
        "date",
        "session_open",
        "session_close",
        "expected_1m_bars",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Session override file is missing: {sorted(missing)}"
        )

    overrides: dict[date, SessionSpec] = {}
    for row in frame.to_dicts():
        if not str(row["date"]).strip():
            continue
        session = date.fromisoformat(
            str(row["date"])
        )
        overrides[session] = SessionSpec(
            session_open=time.fromisoformat(
                str(row["session_open"])
            ),
            session_close=time.fromisoformat(
                str(row["session_close"])
            ),
            expected_1m_bars=int(
                row["expected_1m_bars"]
            ),
        )
    return overrides


def get_session_spec(
    session_date: date,
    overrides: dict[date, SessionSpec] | None = None,
) -> SessionSpec:
    return (overrides or {}).get(
        session_date,
        STANDARD_SESSION,
    )


def is_trading_day(
    value: date,
    holidays: set[date] | None = None,
    overrides: dict[date, SessionSpec] | None = None,
    closed_days: set[date] | None = None,
) -> bool:
    if value in (overrides or {}):
        return True
    return (
        value.weekday() < 5
        and value not in (holidays or set())
        and value not in (closed_days or set())
    )


def session_open(
    session_date: date,
    overrides: dict[date, SessionSpec] | None = None,
) -> datetime:
    spec = get_session_spec(
        session_date,
        overrides,
    )
    return datetime.combine(
        session_date,
        spec.session_open,
        tzinfo=IST,
    )


def session_close(
    session_date: date,
    overrides: dict[date, SessionSpec] | None = None,
) -> datetime:
    spec = get_session_spec(
        session_date,
        overrides,
    )
    return datetime.combine(
        session_date,
        spec.session_close,
        tzinfo=IST,
    )


def expected_1m_timestamps(
    session_date: date,
    overrides: dict[date, SessionSpec] | None = None,
) -> list[datetime]:
    current = session_open(
        session_date,
        overrides,
    )
    end = session_close(
        session_date,
        overrides,
    )
    out: list[datetime] = []
    while current < end:
        out.append(current)
        current += timedelta(minutes=1)
    return out
