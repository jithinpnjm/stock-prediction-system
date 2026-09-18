from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from src.data.calendar import NSECalendar
from src.data.fyers import FyersConfig, create_fyers_client, fetch_window
from src.data.ingest import sha256_file, write_immutable
from src.mlops.manifests import write_dataset_manifest


class FyersBankNiftyDownloader:
    """Canonical append-only downloader for raw FYERS 1m source candles."""

    def __init__(
        self,
        *,
        symbol: str = "NSE:NIFTYBANK-INDEX",
        output_dir: str | Path = "data/raw/fyers",
        manifest_dir: str | Path = "data/raw/fyers_manifests",
        holiday_file: str | Path = "configs/data/nse_holidays.yaml",
        auth_file: str | Path | None = None,
        chunk_days: int = 60,
    ):
        self.output_dir = Path(output_dir)
        self.manifest_dir = Path(manifest_dir)
        self.calendar = NSECalendar.from_yaml(holiday_file)
        self.auth_file = auth_file
        self.config = FyersConfig(
            symbol=symbol,
            chunk_days=chunk_days,
            min_complete_candles=self.calendar.expected_minute_count(),
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    def _existing_days(self) -> set[date]:
        days = set()
        for path in self.output_dir.glob("*.parquet"):
            try:
                days.add(date.fromisoformat(path.stem))
            except ValueError:
                continue
        return days

    def _validate_source_day(self, day: pl.DataFrame, trading_date: date) -> None:
        if day.height != self.config.min_complete_candles:
            raise ValueError(
                f"{trading_date}: expected {self.config.min_complete_candles} "
                f"raw 1m candles, found {day.height}"
            )
        day = self.calendar.filter_source_session(day)
        if day.height != self.config.min_complete_candles:
            raise ValueError(
                f"{trading_date}: source-session filter retained {day.height} "
                f"candles; expected {self.config.min_complete_candles}"
            )
        timestamps = day.sort("timestamp")["timestamp"]
        if timestamps.n_unique() != day.height:
            raise ValueError(f"{trading_date}: duplicate raw timestamps")
        deltas = (
            day.sort("timestamp")
            .with_columns(
                pl.col("timestamp").diff().dt.total_seconds().alias("_delta")
            )
            .filter(pl.col("_delta").is_not_null())
        )
        if deltas.filter(pl.col("_delta") != 60).height:
            raise ValueError(f"{trading_date}: raw candles are not exactly 1 minute apart")

    def download_range(
        self,
        start: date,
        end: date,
    ) -> list[Path]:
        if end < start:
            raise ValueError("end must be on or after start")

        trading_days = self.calendar.trading_days(start, end)
        existing = self._existing_days()
        missing = [d for d in trading_days if d not in existing]

        if not missing:
            return []

        client = create_fyers_client(self.auth_file)
        frame = fetch_window(client, start, end, self.config)
        written: list[Path] = []

        for d in missing:
            day = frame.filter(pl.col("timestamp").dt.date() == d).sort("timestamp")
            if day.is_empty():
                # Do not create a placeholder snapshot. A future run must retry.
                continue

            self._validate_source_day(day, d)

            destination = self.output_dir / f"{d.isoformat()}.parquet"
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite raw snapshot: {destination}")

            write_immutable(day, destination)
            manifest = {
                "dataset_id": f"banknifty_raw_1m_{d.isoformat()}",
                "source": f"fyers:{self.config.symbol}",
                "source_timestamp_semantics": "start",
                "timezone": self.calendar.timezone,
                "trading_date": d.isoformat(),
                "rows": day.height,
                "sha256": sha256_file(destination),
            }
            write_dataset_manifest(
                self.manifest_dir / f"{d.isoformat()}.json",
                manifest,
            )
            written.append(destination)

        return written


def required_missing_days(
    start: date,
    end: date,
    existing_paths: set[date],
    calendar: NSECalendar,
) -> list[date]:
    return [
        d
        for d in calendar.trading_days(start, end)
        if d not in existing_paths
    ]


def _load_state(path: Path) -> set[date]:
    if not path.exists():
        return set()
    return {date.fromisoformat(x) for x in json.loads(path.read_text())}


def _save_state(path: Path, days: set[date]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(d.isoformat() for d in days), indent=2) + "\n")


def mark_no_data_day(path: str | Path, trading_date: date) -> None:
    """Record a confirmed no-data date without creating a fake source snapshot."""
    state_path = Path(path)
    days = _load_state(state_path)
    days.add(trading_date)
    _save_state(state_path, days)
