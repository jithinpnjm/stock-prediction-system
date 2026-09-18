from __future__ import annotations

import argparse
from datetime import date, timedelta

from src.data.downloader import FyersBankNiftyDownloader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download immutable FYERS Bank Nifty 1m source session files"
    )
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--years-back", type=int, default=5)
    parser.add_argument("--auth-file", default=None)
    args = parser.parse_args()

    if args.years_back < 1:
        raise ValueError("--years-back must be >= 1")

    today = date.today()
    start = args.start or (
        today - timedelta(days=365 * args.years_back + args.years_back // 4)
    )
    end = args.end or (today - timedelta(days=1))
    if end < start:
        raise ValueError("end must be on or after start")

    paths = FyersBankNiftyDownloader(auth_file=args.auth_file).download_range(
        start,
        end,
    )
    print(f"Wrote {len(paths)} new session files.")


if __name__ == "__main__":
    main()
