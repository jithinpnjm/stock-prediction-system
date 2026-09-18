from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import polars as pl

from src.common.contracts import SESSION_CLOSE, SESSION_OPEN
from src.data.calendar import load_holidays
from src.data.schemas import expected_bars, validate_schema


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    sessions: int
    duplicate_timestamps: int
    non_monotonic_sessions: int
    invalid_geometry: int
    invalid_price_or_volume: int
    out_of_session: int
    missing_session_bars: int
    unexpected_sessions: tuple[date, ...]

    @property
    def ok(self) -> bool:
        return not any(
            (
                self.duplicate_timestamps,
                self.non_monotonic_sessions,
                self.invalid_geometry,
                self.invalid_price_or_volume,
                self.out_of_session,
                self.missing_session_bars,
                self.unexpected_sessions,
            )
        )

    def to_dict(self) -> dict:
        return asdict(self)


def validate_1m(
    df: pl.DataFrame,
    *,
    holidays_path: str | None = None,
    require_complete_sessions: bool = True,
) -> ValidationReport:
    validate_schema(df, "1m")
    holidays = load_holidays(holidays_path)

    duplicate_timestamps = int(
        df.height - df.select(pl.col("timestamp").n_unique()).item()
    )
    invalid_geometry = df.filter(
        (pl.col("low") > pl.col("high"))
        | (pl.col("open") < pl.col("low"))
        | (pl.col("close") < pl.col("low"))
        | (pl.col("open") > pl.col("high"))
        | (pl.col("close") > pl.col("high"))
    ).height
    invalid_price_or_volume = df.filter(
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

    non_monotonic_sessions = 0
    missing_session_bars = 0
    unexpected_sessions: list[date] = []
    expected = expected_bars("1m")

    for session_df in df.partition_by("session_date", as_dict=False):
        session = session_df.get_column("session_date")[0]
        ts = session_df.get_column("timestamp").to_list()
        if any(b <= a for a, b in zip(ts, ts[1:])):
            non_monotonic_sessions += 1
        if session.weekday() >= 5 or session in holidays:
            unexpected_sessions.append(session)
        elif require_complete_sessions and len(ts) != expected:
            missing_session_bars += abs(expected - len(ts))

    return ValidationReport(
        rows=df.height,
        sessions=df.get_column("session_date").n_unique(),
        duplicate_timestamps=duplicate_timestamps,
        non_monotonic_sessions=non_monotonic_sessions,
        invalid_geometry=invalid_geometry,
        invalid_price_or_volume=invalid_price_or_volume,
        out_of_session=out_of_session,
        missing_session_bars=missing_session_bars,
        unexpected_sessions=tuple(sorted(set(unexpected_sessions))),
    )


def assert_valid_1m(df: pl.DataFrame, **kwargs) -> ValidationReport:
    report = validate_1m(df, **kwargs)
    if not report.ok:
        raise ValueError(f"1m validation failed: {report}")
    return report
