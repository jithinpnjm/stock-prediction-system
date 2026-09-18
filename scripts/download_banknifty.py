from __future__ import annotations

import argparse
from datetime import date

from src.data.downloader import FyersBankNiftyDownloader


def main() -> None:
    parser = argparse.ArgumentParser(description="Download immutable Fyers Bank Nifty 1m session files")
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    downloader = FyersBankNiftyDownloader()
    paths = downloader.download_range(args.start, args.end, overwrite=args.overwrite)
    print(f"Wrote {len(paths)} session files.")


if __name__ == "__main__":
    main()
