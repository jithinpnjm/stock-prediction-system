from __future__ import annotations

import os
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from src.common.contracts import (
    EXPECTED_1M_BARS,
    MARKET_TIMEZONE,
)
from src.data.calendar import (
    get_session_spec,
    is_trading_day,
    load_holidays,
    load_session_overrides,
)
from src.data.fyers import (
    FyersCredentials,
    create_fyers_client,
)
from src.mlops.manifests import (
    dataset_manifest,
    write_manifest,
)


class FyersBankNiftyDownloader:
    def __init__(
        self,
        *,
        symbol: str = "NSE:NIFTYBANK-INDEX",
        resolution: str = "1",
        output_dir: str | Path = "data/raw/fyers",
        manifest_dir: str | Path = (
            "data/raw/fyers_manifests"
        ),
        holiday_file: str | Path = (
            "configs/data/nse_holidays.csv"
        ),
        session_overrides_file: str | Path = (
            "configs/data/session_overrides.csv"
        ),
        chunk_days: int = 60,
        max_attempts: int = 5,
        retry_sleep: float = 5.0,
        inter_chunk_sleep: float = 1.0,
    ):
        self.symbol = symbol
        self.resolution = resolution
        self.output_dir = Path(output_dir)
        self.manifest_dir = Path(manifest_dir)
        self.holidays = load_holidays(
            holiday_file
        )
        self.session_overrides = (
            load_session_overrides(
                session_overrides_file
            )
        )
        self.chunk_days = chunk_days
        self.max_attempts = max_attempts
        self.retry_sleep = retry_sleep
        self.inter_chunk_sleep = (
            inter_chunk_sleep
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.manifest_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _request(
        self,
        client,
        start: date,
        end: date,
    ) -> list[list]:
        payload = {
            "symbol": self.symbol,
            "resolution": self.resolution,
            "date_format": "1",
            "range_from": start.isoformat(),
            "range_to": end.isoformat(),
            "cont_flag": "1",
        }

        for attempt in range(
            1,
            self.max_attempts + 1,
        ):
            response = client.history(
                data=payload
            )

            if not isinstance(response, dict):
                if (
                    attempt
                    == self.max_attempts
                ):
                    raise RuntimeError(
                        "Invalid Fyers response"
                    )
                time.sleep(
                    self.retry_sleep * attempt
                )
                continue

            message = str(
                response.get(
                    "message",
                    response,
                )
            )
            rate_limited = (
                response.get("code")
                == -429
                or (
                    "rate" in message.lower()
                    and "limit"
                    in message.lower()
                )
            )

            if rate_limited:
                if (
                    attempt
                    == self.max_attempts
                ):
                    raise RuntimeError(
                        "Fyers rate limit persisted"
                    )
                time.sleep(
                    max(
                        15.0,
                        self.retry_sleep
                        * attempt,
                    )
                )
                continue

            if response.get("s") == "no_data":
                return []

            if response.get("s") == "ok":
                return response.get(
                    "candles",
                    [],
                )

            if (
                attempt
                == self.max_attempts
            ):
                raise RuntimeError(
                    "Fyers history failed: "
                    + message
                )

            time.sleep(
                self.retry_sleep * attempt
            )

        return []

    @staticmethod
    def _normalize(
        candles: list[list],
    ) -> pl.DataFrame:
        if not candles:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime(
                        time_zone=MARKET_TIMEZONE
                    ),
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                }
            )

        return (
            pl.DataFrame(
                candles,
                schema=[
                    "epoch",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ],
                orient="row",
            )
            .with_columns(
                [
                    pl.from_epoch(
                        pl.col("epoch"),
                        time_unit="s",
                    )
                    .dt.replace_time_zone("UTC")
                    .dt.convert_time_zone(
                        MARKET_TIMEZONE
                    )
                    .alias("timestamp"),
                    pl.col("open").cast(
                        pl.Float64
                    ),
                    pl.col("high").cast(
                        pl.Float64
                    ),
                    pl.col("low").cast(
                        pl.Float64
                    ),
                    pl.col("close").cast(
                        pl.Float64
                    ),
                    pl.col("volume").cast(
                        pl.Float64
                    ),
                ]
            )
            .drop("epoch")
        )

    def download_range(
        self,
        start: date,
        end: date,
    ) -> list[Path]:
        credentials = (
            FyersCredentials
            .from_env_or_file()
        )
        client = create_fyers_client(
            credentials
        )
        written: list[Path] = []

        current = start
        while current <= end:
            chunk_end = min(
                current
                + timedelta(
                    days=self.chunk_days - 1
                ),
                end,
            )
            frame = self._normalize(
                self._request(
                    client,
                    current,
                    chunk_end,
                )
            )

            if not frame.is_empty():
                sessions = sorted(
                    set(
                        frame.get_column(
                            "timestamp"
                        )
                        .dt.date()
                        .to_list()
                    )
                )

                for session in sessions:
                    if not is_trading_day(
                        session,
                        self.holidays,
                        self.session_overrides,
                    ):
                        continue

                    day = (
                        frame.filter(
                            pl.col(
                                "timestamp"
                            ).dt.date()
                            == session
                        )
                        .sort("timestamp")
                    )
                    spec = get_session_spec(
                        session,
                        self.session_overrides,
                    )

                    if day.height != spec.expected_1m_bars:
                        raise RuntimeError(
                            f"{session} returned "
                            f"{day.height} bars; expected "
                            f"{spec.expected_1m_bars} for "
                            "its session calendar entry."
                        )

                    path = (
                        self.output_dir
                        / f"{session.isoformat()}.parquet"
                    )
                    if path.exists():
                        continue

                    tmp = path.with_name(
                        path.name + ".tmp"
                    )
                    day.write_parquet(
                        tmp,
                        compression="zstd",
                    )
                    os.replace(tmp, path)

                    manifest = dataset_manifest(
                        path,
                        schema={
                            key: str(value)
                            for key, value
                            in day.schema.items()
                        },
                        row_count=day.height,
                        source=(
                            f"fyers:{self.symbol}"
                        ),
                        version="raw-1m-v1",
                    )
                    write_manifest(
                        manifest,
                        self.manifest_dir
                        / f"{session.isoformat()}.json",
                    )
                    written.append(path)

            current = (
                chunk_end
                + timedelta(days=1)
            )
            time.sleep(
                self.inter_chunk_sleep
            )

        return written
