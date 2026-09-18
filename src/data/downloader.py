from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from src.common.contracts import MARKET_TIMEZONE
from src.data.fyers import FyersCredentials, create_fyers_client

IST = ZoneInfo(MARKET_TIMEZONE)
UTC = timezone.utc


class FyersBankNiftyDownloader:
    def __init__(
        self,
        *,
        symbol: str = "NSE:NIFTYBANK-INDEX",
        resolution: str = "1",
        output_dir: str | Path = "data/raw/fyers",
        chunk_days: int = 60,
        max_attempts: int = 5,
        retry_sleep: float = 5.0,
        inter_chunk_sleep: float = 1.0,
    ):
        self.symbol = symbol
        self.resolution = resolution
        self.output_dir = Path(output_dir)
        self.chunk_days = chunk_days
        self.max_attempts = max_attempts
        self.retry_sleep = retry_sleep
        self.inter_chunk_sleep = inter_chunk_sleep
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.no_data_path = self.output_dir / "no_data_days.json"

    def _load_no_data_days(self) -> set[date]:
        if not self.no_data_path.exists():
            return set()
        values = json.loads(self.no_data_path.read_text(encoding="utf-8"))
        return {date.fromisoformat(v) for v in values}

    def _save_no_data_days(self, days: set[date]) -> None:
        self.no_data_path.write_text(
            json.dumps(sorted(d.isoformat() for d in days), indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _to_ist(ts: int | float) -> datetime:
        return datetime.fromtimestamp(ts, tz=UTC).astimezone(IST)

    def _request(self, client, start: date, end: date) -> list[list]:
        request = {
            "symbol": self.symbol,
            "resolution": self.resolution,
            "date_format": "1",
            "range_from": start.isoformat(),
            "range_to": end.isoformat(),
            "cont_flag": "1",
        }
        for attempt in range(1, self.max_attempts + 1):
            response = client.history(data=request)
            if not isinstance(response, dict):
                if attempt == self.max_attempts:
                    raise RuntimeError(f"Invalid Fyers response: {response!r}")
                time.sleep(self.retry_sleep * attempt)
                continue
            code = response.get("code")
            if code == -429 or str(response.get("s")).lower() == "error" and "limit" in str(response).lower():
                if attempt == self.max_attempts:
                    raise RuntimeError(f"Fyers rate limit after {attempt} attempts")
                time.sleep(max(15.0, self.retry_sleep * attempt))
                continue
            if response.get("s") == "no_data":
                return []
            if response.get("s") == "ok":
                return response.get("candles", [])
            if attempt == self.max_attempts:
                raise RuntimeError(
                    f"Fyers history failed [{code}]: {response.get('message', response)}"
                )
            time.sleep(self.retry_sleep * attempt)
        return []

    def download_range(
        self,
        start: date,
        end: date,
        *,
        overwrite: bool = False,
    ) -> list[Path]:
        no_data = self._load_no_data_days()
        credentials = FyersCredentials.from_env_or_file()
        client = create_fyers_client(credentials)
        written: list[Path] = []

        current = start
        while current <= end:
            chunk_end = min(
                current + timedelta(days=self.chunk_days - 1),
                end,
            )
            candles = self._request(client, current, chunk_end)

            if candles:
                df = pl.DataFrame(
                    candles,
                    schema=["epoch", "open", "high", "low", "close", "volume"],
                    orient="row",
                ).with_columns(
                    pl.from_epoch(pl.col("epoch"), time_unit="s")
                    .dt.replace_time_zone("UTC")
                    .dt.convert_time_zone(MARKET_TIMEZONE)
                    .alias("timestamp")
                ).drop("epoch")

                for session in sorted(set(df.get_column("timestamp").dt.date().to_list())):
                    day_df = (
                        df.filter(pl.col("timestamp").dt.date() == session)
                        .sort("timestamp")
                    )
                    path = self.output_dir / f"{session.isoformat()}.parquet"
                    if path.exists() and not overwrite:
                        continue
                    tmp = path.with_suffix(".parquet.tmp")
                    day_df.write_parquet(tmp, compression="zstd")
                    os.replace(tmp, path)
                    written.append(path)
            else:
                d = current
                while d <= chunk_end:
                    if d.weekday() < 5:
                        no_data.add(d)
                    d += timedelta(days=1)
                self._save_no_data_days(no_data)

            current = chunk_end + timedelta(days=1)
            time.sleep(self.inter_chunk_sleep)

        return written
