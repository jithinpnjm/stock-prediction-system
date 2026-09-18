from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta

import polars as pl

from .calendar import NSECalendar
from .schemas import DataContract, validate_required_schema


def _report(
    name: str,
    passed: bool,
    count: int = 0,
    details: str = "",
) -> dict[str, object]:
    return {"gate": name, "passed": passed, "count": count, "details": details}


def validate_ohlcv(
    df: pl.DataFrame,
    *,
    calendar: NSECalendar,
    contract: DataContract | None = None,
) -> list[dict[str, object]]:
    contract = contract or DataContract()
    validate_required_schema(df)
    if df.is_empty():
        return [_report("non_empty", False, details="dataset is empty")]

    reports: list[dict[str, object]] = []
    nulls = df.null_count().sum_horizontal().item()
    reports.append(_report("no_nulls", nulls == 0, nulls))

    duplicate_count = df.height - df.select("timestamp").unique().height
    reports.append(_report("no_duplicate_timestamps", duplicate_count == 0, duplicate_count))

    sorted_ok = df.select(pl.col("timestamp").is_sorted()).item()
    reports.append(_report("globally_sorted", bool(sorted_ok)))

    invalid = df.filter(
        (pl.col("high") < pl.col("low"))
        | (pl.col("open") < pl.col("low"))
        | (pl.col("open") > pl.col("high"))
        | (pl.col("close") < pl.col("low"))
        | (pl.col("close") > pl.col("high"))
        | (pl.col("high") < 0)
        | (pl.col("low") < 0)
        | (pl.col("open") < 0)
        | (pl.col("close") < 0)
        | (pl.col("volume") < 0)
    )
    reports.append(_report("ohlcv_geometry", invalid.is_empty(), invalid.height))

    session_df = calendar.filter_session(df)
    outside = df.height - session_df.height
    reports.append(_report("session_boundaries", outside == 0, outside))

    expected_gap_rows = 0
    unexpected_days: list[str] = []
    min_date = df.select(pl.col("timestamp").min().dt.date()).item()
    max_date = df.select(pl.col("timestamp").max().dt.date()).item()
    for d in calendar.trading_days(min_date, max_date):
        day = session_df.filter(pl.col("timestamp").dt.date() == d)
        if day.is_empty():
            unexpected_days.append(d.isoformat())
            continue
        expected = calendar.expected_minute_count()
        if day.height < expected * 0.98:
            expected_gap_rows += expected - day.height
    reports.append(
        _report(
            "session_completeness",
            not unexpected_days and expected_gap_rows == 0,
            expected_gap_rows,
            f"missing_sessions={unexpected_days}",
        )
    )

    gaps = (
        session_df.sort("timestamp")
        .with_columns(pl.col("timestamp").diff().dt.total_seconds().alias("gap_seconds"))
        .filter(pl.col("gap_seconds") > 60)
    )
    reports.append(_report("unexpected_intraday_gaps", gaps.is_empty(), gaps.height))

    stale = session_df.filter(
        (pl.col("open") == pl.col("high"))
        & (pl.col("high") == pl.col("low"))
        & (pl.col("low") == pl.col("close"))
    )
    # Stale bars are diagnostic rather than an automatic failure.
    reports.append(_report("stale_candle_diagnostic", True, stale.height))

    return reports


def assert_valid(
    df: pl.DataFrame,
    *,
    calendar: NSECalendar,
    contract: DataContract | None = None,
) -> list[dict[str, object]]:
    reports = validate_ohlcv(df, calendar=calendar, contract=contract)
    critical = [r for r in reports if r["gate"] != "stale_candle_diagnostic" and not r["passed"]]
    if critical:
        raise ValueError(
            "Data quality validation failed: "
            + "; ".join(f"{r['gate']}({r['count']})" for r in critical)
        )
    return reports
