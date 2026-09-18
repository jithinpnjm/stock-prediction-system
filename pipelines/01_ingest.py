import os
import sys

import polars as pl


def run():
    print("Running pipeline step: 01_ingest.py")

    # User will override this path dynamically based on Nebius storage mounts
    RAW_DATA_PATH = os.environ.get("RAW_1M_DATA_PATH", "data/raw/1m_data.parquet")

    if not os.path.exists(RAW_DATA_PATH):
        print(f"ERROR: Raw data not found at {RAW_DATA_PATH}.")
        print("Please mount your data or set RAW_1M_DATA_PATH environment variable.")
        print(
            "To generate synthetic mock data for testing, run: python tests/data_quality/mock_data.py"
        )
        sys.exit(1)

    try:
        # We enforce reading as Parquet as described in the architecture plan
        df = pl.read_parquet(RAW_DATA_PATH)

        # Validate schema basics
        required_cols = {"datetime", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(
                f"Data is missing required columns. Needs: {required_cols}"
            )

        print(f"Successfully ingested {df.height} rows from {RAW_DATA_PATH}")

    except Exception as e:
        print(f"Data ingestion failed: {e!s}")
        sys.exit(1)


if __name__ == "__main__":
    run()
