from __future__ import annotations

from dataclasses import dataclass
from datetime import date,datetime,time,timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import yaml


@dataclass(frozen=True)
class Session:
    trading_date:date
    open_time:datetime
    close_time:datetime


class NSECalendar:
    def __init__(
        self,
        holidays=None,
        timezone="Asia/Kolkata",
        session_open:time=time(9,15),
        session_close:time=time(15,30),
    ):
        self.holidays=set(holidays or [])
        self.timezone=timezone
        self.session_open=session_open
        self.session_close=session_close

    @classmethod
    def from_yaml(cls,path:str|Path)->"NSECalendar":
        payload=yaml.safe_load(Path(path).read_text()) or {}
        holidays={date.fromisoformat(x) for x in payload.get("holidays",[])}
        return cls(
            holidays=holidays,
            timezone=payload.get("timezone","Asia/Kolkata"),
        )

    def is_trading_day(self,d:date)->bool:
        return d.weekday()<5 and d not in self.holidays

    def session(self,d:date)->Session:
        if not self.is_trading_day(d):
            raise ValueError(f"{d} is not a trading day")
        tz=ZoneInfo(self.timezone)
        return Session(
            d,
            datetime.combine(d,self.session_open,tzinfo=tz),
            datetime.combine(d,self.session_close,tzinfo=tz),
        )

    def expected_minute_count(self)->int:
        return int(
            (
                datetime.combine(date.today(),self.session_close)
                -datetime.combine(date.today(),self.session_open)
            ).total_seconds()/60
        )

    def trading_days(self,start:date,end:date)->list[date]:
        out=[]
        d=start
        while d<=end:
            if self.is_trading_day(d):
                out.append(d)
            d+=timedelta(days=1)
        return out

    def filter_session(self,df:pl.DataFrame,timestamp_col="timestamp")->pl.DataFrame:
        if df.is_empty():
            return df
        dtype=df[timestamp_col].dtype
        if isinstance(dtype,pl.Datetime) and dtype.time_zone:
            local=pl.col(timestamp_col).dt.convert_time_zone(self.timezone)
        else:
            local=pl.col(timestamp_col).cast(
                pl.Datetime(time_zone=None),strict=False
            ).dt.replace_time_zone(self.timezone)

        midnight=local.dt.truncate("1d")
        open_dt=midnight+pl.duration(
            minutes=self.session_open.hour*60+self.session_open.minute
        )
        close_dt=midnight+pl.duration(
            minutes=self.session_close.hour*60+self.session_close.minute
        )
        return df.filter((local>open_dt)&(local<=close_dt))
