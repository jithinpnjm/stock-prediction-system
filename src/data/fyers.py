from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FyersCredentials:
    client_id: str
    access_token: str

    @classmethod
    def from_env_or_file(cls, path: str | Path = "secrets/fyers_auth.json") -> "FyersCredentials":
        client_id = os.getenv("FYERS_CLIENT_ID")
        access_token = os.getenv("FYERS_ACCESS_TOKEN")
        if client_id and access_token:
            return cls(client_id=client_id, access_token=access_token)

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                "Fyers credentials not found. Set FYERS_CLIENT_ID/FYERS_ACCESS_TOKEN "
                "or create a local secrets/fyers_auth.json file."
            )
        payload = json.loads(p.read_text(encoding="utf-8"))
        client_id = payload.get("client_id") or payload.get("FYERS_CLIENT_ID")
        access_token = payload.get("access_token") or payload.get("token") or payload.get("FYERS_ACCESS_TOKEN")
        if not client_id or not access_token:
            raise ValueError("Fyers auth file must contain client_id and access_token")
        return cls(client_id=client_id, access_token=access_token)


def create_fyers_client(credentials: FyersCredentials):
    try:
        from fyers_apiv3 import fyersModel
    except ImportError as exc:
        raise ImportError("Install fyers-apiv3 to use the Fyers downloader") from exc
    return fyersModel.FyersModel(
        client_id=credentials.client_id,
        token=credentials.access_token,
        is_async=False,
        log_path="",
    )
