from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import yaml


@dataclass(frozen=True)
class Session:
    trading_date: date
    open_time: datetime
    close_time: datetime


class NSECalendar:
    def __init__(
        self,
        holidays=None,
        timezone: str = "Asia/Kolkata",
        session_open: time = time(9, 15),
        session_close: time = time(15, 30),
    ):
        self.holidays = set(holidays or [])
        self.timezone = timezone
        self.session_open = session_open
        self.session_close = session_close

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NSECalendar":
        payload = yaml.safe_load(Path(path).read_text()) or {}
        holidays = {date.fromisoformat(x) for x in payload.get("holidays", [])}
        session_open = time.fromisoformat(payload.get("session_open", "09:15"))
        session_close = time.fromisoformat(payload.get("session_close", "15:30"))
        return cls(
            holidays=holidays,
            timezone=payload.get("timezone", "Asia/Kolkata"),
            session_open=session_open,
            session_close=session_close,
        )

    def is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self.holidays

    def session(self, d: date) -> Session:
        if not self.is_trading_day(d):
            raise ValueError(f"{d} is not a trading day")
        tz = ZoneInfo(self.timezone)
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

    def _local_timestamp(self, df: pl.DataFrame, timestamp_col: str) -> pl.Expr:
        dtype = df[timestamp_col].dtype
        source_timezone = getattr(dtype, "time_zone", None)

        if source_timezone:
            # Convert to market timezone first, then remove timezone metadata so
            # Polars time-part extraction operates on the local wall clock.
            return (
                pl.col(timestamp_col)
                .dt.convert_time_zone(self.timezone)
                .dt.replace_time_zone(None)
            )

        return pl.col(timestamp_col).cast(
            pl.Datetime(time_zone=None),
            strict=False,
        )

    def _session_minutes(self, df: pl.DataFrame, timestamp_col: str) -> pl.Expr:
        """Return market-local HH:MM as minutes since midnight.

        Keep the timezone-aware expression intact through conversion. In
        particular, do not strip timezone metadata before extracting the clock;
        that can cause Polars versions to reinterpret the instant in UTC.
        """
        dtype = df[timestamp_col].dtype
        source_timezone = getattr(dtype, "time_zone", None)
        local = pl.col(timestamp_col)
        if source_timezone:
            local = local.dt.convert_time_zone(self.timezone)
        else:
            local = local.cast(pl.Datetime(time_zone=None), strict=False)
        return (
            local.dt.strftime("%H").cast(pl.Int32) * 60
            + local.dt.strftime("%M").cast(pl.Int32)
        )

    def filter_session(
        self,
        df: pl.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> pl.DataFrame:
        """Filter canonical availability timestamps: (open, close]."""
        if df.is_empty():
            return df
        minutes = self._session_minutes(df, timestamp_col)
        open_minute = self.session_open.hour * 60 + self.session_open.minute
        close_minute = self.session_close.hour * 60 + self.session_close.minute
        return df.filter((minutes > open_minute) & (minutes <= close_minute))

    def filter_source_session(
        self,
        df: pl.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> pl.DataFrame:
        """Filter source candle-start timestamps: [open, close)."""
        if df.is_empty():
            return df
        minutes = self._session_minutes(df, timestamp_col)
        open_minute = self.session_open.hour * 60 + self.session_open.minute
        close_minute = self.session_close.hour * 60 + self.session_close.minute
        return df.filter((minutes >= open_minute) & (minutes < close_minute))
