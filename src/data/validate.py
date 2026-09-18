from __future__ import annotations

import polars as pl

from .calendar import NSECalendar
from .schemas import DataContract,validate_required_schema


def _report(name,passed,count=0,details=""):
    return {"gate":name,"passed":bool(passed),"count":int(count),"details":details}


def validate_ohlcv(
    df:pl.DataFrame,
    *,
    calendar:NSECalendar,
    contract:DataContract|None=None,
):
    contract=contract or DataContract()
    validate_required_schema(df)
    if df.is_empty():
        return [_report("non_empty",False,details="dataset is empty")]

    reports=[]
    nulls=int(df.null_count().sum_horizontal().item())
    reports.append(_report("no_nulls",nulls==0,nulls))

    duplicates=df.height-df.select("timestamp").unique().height
    reports.append(_report("no_duplicate_timestamps",duplicates==0,duplicates))

    sorted_ok=bool(df.select(pl.col("timestamp").is_sorted()).item())
    reports.append(_report("sorted",sorted_ok,0,"global timestamp ordering"))

    invalid=df.filter(
        (pl.col("high")<pl.col("low"))|(pl.col("open")<pl.col("low"))|
        (pl.col("open")>pl.col("high"))|(pl.col("close")<pl.col("low"))|
        (pl.col("close")>pl.col("high"))|(pl.col("open")<0)|
        (pl.col("high")<0)|(pl.col("low")<0)|(pl.col("close")<0)|
        (pl.col("volume")<0)
    )
    reports.append(_report("ohlcv_geometry",invalid.is_empty(),invalid.height))

    session=calendar.filter_session(df).sort("timestamp")
    outside=df.height-session.height
    reports.append(_report("session_boundaries",outside==0,outside))

    if session.is_empty():
        reports.append(_report("session_completeness",False,0,"no in-session rows"))
        reports.append(_report("unexpected_intraday_gaps",False,0,"no in-session rows"))
        reports.append(_report("stale_candle_diagnostic",True,0))
        return reports

    missing_sessions=[]
    bad_day_counts=0
    expected=calendar.expected_minute_count()
    min_date=session.select(pl.col("timestamp").min().dt.date()).item()
    max_date=session.select(pl.col("timestamp").max().dt.date()).item()

    for d in calendar.trading_days(min_date,max_date):
        day=session.filter(pl.col("timestamp").dt.date()==d)
        if day.is_empty():
            missing_sessions.append(d.isoformat())
        elif day.height!=expected:
            bad_day_counts+=abs(expected-day.height)

    reports.append(_report(
        "session_completeness",
        not missing_sessions and bad_day_counts==0,
        bad_day_counts,
        f"missing_sessions={missing_sessions}",
    ))

    gap_count=0
    for d in calendar.trading_days(min_date,max_date):
        day=session.filter(pl.col("timestamp").dt.date()==d).sort("timestamp")
        if day.height>1:
            gap_count+=int(
                day.with_columns(
                    pl.col("timestamp").diff().dt.total_seconds().alias("_gap")
                ).filter(pl.col("_gap")>60).height
            )
    reports.append(_report("unexpected_intraday_gaps",gap_count==0,gap_count))

    stale=session.filter(
        (pl.col("open")==pl.col("high"))&
        (pl.col("high")==pl.col("low"))&
        (pl.col("low")==pl.col("close"))
    )
    reports.append(_report("stale_candle_diagnostic",True,stale.height))
    return reports


def assert_valid(
    df:pl.DataFrame,
    *,
    calendar:NSECalendar,
    contract:DataContract|None=None,
):
    reports=validate_ohlcv(df,calendar=calendar,contract=contract)
    critical=[
        r for r in reports
        if r["gate"]!="stale_candle_diagnostic" and not r["passed"]
    ]
    if critical:
        raise ValueError(
            "Data quality validation failed: "+
            "; ".join(f"{r['gate']}({r['count']})" for r in critical)
        )
    return reports
