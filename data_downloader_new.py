"""CLI compatibility wrapper for the canonical FYERS downloader.

Prefer `python scripts/download_banknifty.py` for new workflows.
"""

from __future__ import annotations

import argparse
from datetime import date

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

    today = date.today()
    start = args.start or (today - timedelta(days=365 * args.years_back + args.years_back // 4))
    end = args.end or (today - timedelta(days=1))
    paths = FyersBankNiftyDownloader(auth_file=args.auth_file).download_range(start, end)
    print(f"Wrote {len(paths)} session files.")


if __name__ == "__main__":
    main()
