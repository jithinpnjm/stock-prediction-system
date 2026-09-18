from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable

import polars as pl
import yaml


@dataclass(frozen=True)
class Session:
    trading_date: date
    open_time: datetime
    close_time: datetime


class NSECalendar:
    """Minimal deterministic NSE cash-session calendar.

    Holidays are supplied explicitly. This avoids silently treating every
    weekday as a trading day when the exchange calendar changes.
    """

    def __init__(
        self,
        holidays: Iterable[date] | None = None,
        timezone: str = "Asia/Kolkata",
        session_open: time = time(9, 15),
        session_close: time = time(15, 30),
    ) -> None:
        self.holidays = set(holidays or [])
        self.timezone = timezone
        self.session_open = session_open
        self.session_close = session_close

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NSECalendar":
        payload = yaml.safe_load(Path(path).read_text()) or {}
        holidays = {
            date.fromisoformat(x) for x in payload.get("holidays", [])
        }
        return cls(
            holidays=holidays,
            timezone=payload.get("timezone", "Asia/Kolkata"),
        )

    def is_holiday(self, d: date) -> bool:
        return d.weekday() >= 5 or d in self.holidays

    def is_trading_day(self, d: date) -> bool:
        return not self.is_holiday(d)

    def session(self, d: date) -> Session:
        if not self.is_trading_day(d):
            raise ValueError(f"{d} is not a trading day")
        tz = __import__("zoneinfo").ZoneInfo(self.timezone)
        return Session(
            d,
            datetime.combine(d, self.session_open, tzinfo=tz),
            datetime.combine(d, self.session_close, tzinfo=tz),
        )

    def expected_minute_count(self) -> int:
        return int(
            (
                datetime.combine(date.today(), self.session_close)
                - datetime.combine(date.today(), self.session_open)
            ).total_seconds()
            / 60
        )

    def trading_days(self, start: date, end: date) -> list[date]:
        out = []
        d = start
        while d <= end:
            if self.is_trading_day(d):
                out.append(d)
            d += timedelta(days=1)
        return out

    def filter_session(self, df: pl.DataFrame, timestamp_col: str = "timestamp") -> pl.DataFrame:
        if df.is_empty():
            return df
        local = pl.col(timestamp_col).dt.convert_time_zone(self.timezone)
        minutes = local.dt.hour() * 60 + local.dt.minute()
        open_min = self.session_open.hour * 60 + self.session_open.minute
        close_min = self.session_close.hour * 60 + self.session_close.minute
        return df.filter((minutes >= open_min) & (minutes < close_min))


def load_holidays(path: str | Path) -> set[date]:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    return {date.fromisoformat(x) for x in payload.get("holidays", [])}
