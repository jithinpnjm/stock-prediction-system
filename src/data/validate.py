from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from src.common.contracts import MARKET_TIMEZONE, SESSION_CLOSE, SESSION_OPEN
from src.data.calendar import load_holidays
from src.data.schemas import expected_bars, validate_schema


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    sessions: int
    duplicate_rows: int
    invalid_geometry: int
    invalid_price: int
    out_of_session: int
    duplicate_timestamps: int
    missing_session_bars: int
    unexpected_sessions: tuple[date, ...]

    @property
    def ok(self) -> bool:
        return not any(
            (
                self.duplicate_rows,
                self.invalid_geometry,
                self.invalid_price,
                self.out_of_session,
                self.duplicate_timestamps,
                self.missing_session_bars,
                self.unexpected_sessions,
            )
        )


def validate_1m(
    df: pl.DataFrame,
    *,
    holidays_path: str | None = None,
    require_complete_sessions: bool = True,
) -> ValidationReport:
    validate_schema(df, "1m")
    holidays = load_holidays(holidays_path)

    duplicate_rows = df.height - df.select(pl.col("timestamp").n_unique()).item()
    duplicate_timestamps = duplicate_rows

    invalid_geometry = df.filter(
        (pl.col("low") > pl.col("high"))
        | (pl.col("open") < pl.col("low"))
        | (pl.col("close") < pl.col("low"))
        | (pl.col("open") > pl.col("high"))
        | (pl.col("close") > pl.col("high"))
    ).height

    invalid_price = df.filter(
        (pl.col("open") < 0)
        | (pl.col("high") < 0)
        | (pl.col("low") < 0)
        | (pl.col("close") < 0)
        | (pl.col("volume") < 0)
    ).height

    out_of_session = df.filter(
        ~(
            (pl.col("timestamp").dt.time() >= SESSION_OPEN)
            & (pl.col("timestamp").dt.time() < SESSION_CLOSE)
        )
    ).height

    unexpected_sessions: list[date] = []
    missing_session_bars = 0
    grouped = df.group_by("session_date").agg(
        [
            pl.len().alias("count"),
            pl.col("timestamp").n_unique().alias("unique_ts"),
        ]
    )

    expected = expected_bars("1m")
    for row in grouped.iter_rows(named=True):
        session = row["session_date"]
        count = row["count"]
        if session.weekday() >= 5 or session in holidays:
            unexpected_sessions.append(session)
        elif require_complete_sessions and count != expected:
            missing_session_bars += abs(expected - count)

    # Strict monotonicity across each session.
    unsorted = (
        df.group_by("session_date", maintain_order=True)
        .agg(pl.col("timestamp").diff().drop_nulls().le(pl.duration(minutes=0)).any().alias("bad"))
        .filter(pl.col("bad"))
        .height
    )
    duplicate_rows += unsorted

    return ValidationReport(
        rows=df.height,
        sessions=grouped.height,
        duplicate_rows=duplicate_rows,
        invalid_geometry=invalid_geometry,
        invalid_price=invalid_price,
        out_of_session=out_of_session,
        duplicate_timestamps=duplicate_timestamps,
        missing_session_bars=missing_session_bars,
        unexpected_sessions=tuple(sorted(set(unexpected_sessions))),
    )


def assert_valid_1m(df: pl.DataFrame, **kwargs: object) -> ValidationReport:
    report = validate_1m(df, **kwargs)
    if not report.ok:
        raise ValueError(f"1m validation failed: {report}")
    return report
