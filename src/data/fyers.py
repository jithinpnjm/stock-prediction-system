from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import polars as pl


@dataclass(frozen=True)
class FyersConfig:
    symbol: str = "NSE:NIFTYBANK-INDEX"
    resolution: str = "1"
    chunk_days: int = 60
    max_attempts: int = 5
    retry_base_seconds: int = 5
    inter_chunk_sleep: float = 1.5
    min_complete_candles: int = 375


def load_credentials(path: str | Path | None = None) -> dict[str, str]:
    """Load local Fyers credentials without ever requiring them in Git."""
    payload_path = Path(
        path or os.getenv("FYERS_AUTH_FILE", "secrets/fyers_auth.json")
    )
    if payload_path.exists():
        payload = json.loads(payload_path.read_text())
        return {
            "client_id": payload["client_id"],
            "access_token": payload["access_token"],
        }
    client_id = os.getenv("FYERS_CLIENT_ID")
    token = os.getenv("FYERS_ACCESS_TOKEN")
    if not client_id or not token:
        raise RuntimeError(
            "Fyers credentials not found. Use secrets/fyers_auth.json locally "
            "or FYERS_CLIENT_ID/FYERS_ACCESS_TOKEN environment variables."
        )
    return {"client_id": client_id, "access_token": token}


def create_fyers_client(path: str | Path | None = None):
    try:
        from fyers_apiv3 import fyersModel
    except ImportError as exc:
        raise RuntimeError(
            "Install the ingest extra: pip install -e '.[ingest]'"
        ) from exc
    creds = load_credentials(path)
    return fyersModel.FyersModel(
        client_id=creds["client_id"],
        token=creds["access_token"],
        is_async=False,
        log_path="",
    )


def _response_frame(response: dict) -> pl.DataFrame:
    candles = response.get("candles") or []
    if not candles:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime(time_zone="Asia/Kolkata"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )
    raw = pd.DataFrame(
        candles,
        columns=["epoch", "open", "high", "low", "close", "volume"],
    )
    raw["timestamp"] = (
        pd.to_datetime(raw["epoch"], unit="s", utc=True)
        .dt.tz_convert("Asia/Kolkata")
    )
    return pl.from_pandas(
        raw[["timestamp", "open", "high", "low", "close", "volume"]]
    ).with_columns(
        pl.col("timestamp").cast(pl.Datetime(time_zone="Asia/Kolkata"))
    )


def fetch_window(
    client,
    start: date,
    end: date,
    config: FyersConfig | None = None,
) -> pl.DataFrame:
    cfg = config or FyersConfig()
    if cfg.chunk_days < 1:
        raise ValueError("chunk_days must be >= 1")

    all_frames: list[pl.DataFrame] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=cfg.chunk_days - 1), end)
        payload = {
            "symbol": cfg.symbol,
            "resolution": cfg.resolution,
            "date_format": "1",
            "range_from": cursor.isoformat(),
            "range_to": chunk_end.isoformat(),
            "cont_flag": "1",
        }
        last_error = None
        for attempt in range(cfg.max_attempts):
            try:
                response = client.history(data=payload)
                if not isinstance(response, dict):
                    raise RuntimeError(f"invalid Fyers response: {response!r}")
                code = response.get("code")
                message = str(response.get("message", ""))
                if code == -429 or "limit" in message.lower():
                    time.sleep(cfg.retry_base_seconds * (2**attempt))
                    continue
                if response.get("s") == "ok":
                    frame = _response_frame(response)
                    if not frame.is_empty():
                        all_frames.append(frame)
                    last_error = None
                    break
                if response.get("s") == "no_data":
                    last_error = None
                    break
                last_error = RuntimeError(
                    f"Fyers error code={code}: {response.get('message', response)}"
                )
                if code in {400, 494, -101, -300}:
                    break
            except Exception as exc:
                last_error = exc
            time.sleep(cfg.retry_base_seconds * (2**attempt))
        if last_error is not None:
            raise last_error
        time.sleep(cfg.inter_chunk_sleep)
        cursor = chunk_end + timedelta(days=1)

    if not all_frames:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime(time_zone="Asia/Kolkata"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )
    return (
        pl.concat(all_frames)
        .unique(subset=["timestamp"], keep="last")
        .sort("timestamp")
    )
