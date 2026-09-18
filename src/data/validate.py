from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta

import polars as pl

from src.common.contracts import (
    SESSION_CLOSE,
    SESSION_OPEN,
)
from src.data.calendar import (
    get_session_spec,
    is_trading_day,
    load_closed_days,
    load_holidays,
    load_session_overrides,
)
from src.data.schemas import (
    validate_schema,
)


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    sessions: int
    duplicate_timestamps: int
    non_monotonic_sessions: int
    timestamp_gap_count: int
    invalid_geometry: int
    invalid_price_or_volume: int
    out_of_session: int
    invalid_minute_alignment: int
    missing_session_bars: int
    missing_trading_sessions: int
    unexpected_sessions: tuple[date, ...]

    @property
    def ok(self) -> bool:
        return not any(
            (
                self.duplicate_timestamps,
                self.non_monotonic_sessions,
                self.timestamp_gap_count,
                self.invalid_geometry,
                self.invalid_price_or_volume,
                self.out_of_session,
                self.invalid_minute_alignment,
                self.missing_session_bars,
                self.missing_trading_sessions,
                self.unexpected_sessions,
            )
        )

    def to_dict(self) -> dict:
        return asdict(self)


def validate_1m(
    df: pl.DataFrame,
    *,
    holidays_path: str | None = None,
    closed_days_path: str | None = None,
    session_overrides_path: str | None = None,
    require_complete_sessions: bool = True,
) -> ValidationReport:
    validate_schema(df, "1m")

    holidays = load_holidays(
        holidays_path
    )
    closed_days = load_closed_days(
        closed_days_path
    )
    overrides = load_session_overrides(
        session_overrides_path
    )

    duplicate_timestamps = int(
        df.height
        - df.select(
            pl.col("timestamp").n_unique()
        ).item()
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

    invalid_minute_alignment = df.filter(
        (pl.col("timestamp").dt.second() != 0)
        | (pl.col("timestamp").dt.microsecond() != 0)
    ).height

    non_monotonic_sessions = 0
    timestamp_gap_count = 0
    out_of_session = 0
    missing_session_bars = 0
    unexpected_sessions: list[date] = []

    session_dates = sorted(
        set(
            df.get_column(
                "session_date"
            ).to_list()
        )
    )
    present = set(session_dates)

    for session_df in df.partition_by(
        "session_date",
        as_dict=False,
    ):
        session = session_df.get_column(
            "session_date"
        )[0]
        timestamps = session_df.get_column(
            "timestamp"
        ).to_list()
        spec = get_session_spec(
            session,
            overrides,
        )

        if any(
            later <= earlier
            for earlier, later in zip(
                timestamps,
                timestamps[1:],
            )
        ):
            non_monotonic_sessions += 1

        timestamp_gap_count += sum(
            (later - earlier).total_seconds()
            != 60
            for earlier, later in zip(
                timestamps,
                timestamps[1:],
            )
        )

        out_of_session += sum(
            not (
                spec.session_open
                <= timestamp.timetz().replace(
                    tzinfo=None
                )
                < spec.session_close
            )
            for timestamp in timestamps
        )

        expected_for_day = is_trading_day(
            session,
            holidays,
            overrides,
            closed_days,
        )
        if not expected_for_day:
            unexpected_sessions.append(
                session
            )
        elif (
            require_complete_sessions
            and len(timestamps)
            != spec.expected_1m_bars
        ):
            missing_session_bars += abs(
                spec.expected_1m_bars
                - len(timestamps)
            )

    missing_trading_sessions = 0
    if session_dates:
        current = session_dates[0]
        last = session_dates[-1]
        while current <= last:
            if (
                is_trading_day(
                    current,
                    holidays,
                    overrides,
                    closed_days,
                )
                and current not in present
            ):
                missing_trading_sessions += 1
            current += timedelta(days=1)

    return ValidationReport(
        rows=df.height,
        sessions=len(session_dates),
        duplicate_timestamps=duplicate_timestamps,
        non_monotonic_sessions=non_monotonic_sessions,
        timestamp_gap_count=timestamp_gap_count,
        invalid_geometry=invalid_geometry,
        invalid_price_or_volume=invalid_price_or_volume,
        out_of_session=out_of_session,
        invalid_minute_alignment=invalid_minute_alignment,
        missing_session_bars=missing_session_bars,
        missing_trading_sessions=missing_trading_sessions,
        unexpected_sessions=tuple(
            sorted(set(unexpected_sessions))
        ),
    )


def assert_valid_1m(
    df: pl.DataFrame,
    **kwargs,
) -> ValidationReport:
    report = validate_1m(df, **kwargs)
    if not report.ok:
        raise ValueError(
            f"1m validation failed: {report}"
        )
    return report
