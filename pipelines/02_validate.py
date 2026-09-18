import polars as pl
import sys
import os

def run():
    print("Running pipeline step: 02_validate.py")
    
    RAW_DATA_PATH = os.environ.get("RAW_1M_DATA_PATH", "data/raw/1m_data.parquet")
    
    try:
        df = pl.read_parquet(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Could not read {RAW_DATA_PATH}. Run 01_ingest.py first.")
        sys.exit(1)
        
    print(f"Running data quality gates on {df.height} rows...")
    
    # Gate 1: Monotonic and duplicate timestamps
    if df.height != df.select(pl.col("datetime").n_unique()).item():
        print("FAILED GATE: Duplicate timestamps found.")
        sys.exit(1)
        
    is_sorted = df.select(pl.col("datetime").is_sorted()).item()
    if not is_sorted:
        print("FAILED GATE: Timestamps are not strictly monotonic/sorted.")
        sys.exit(1)
        
    # Gate 2: OHLC Geometry Logic (Low <= Open/Close <= High)
    invalid_geometry = df.filter(
        (pl.col("low") > pl.col("high")) |
        (pl.col("open") < pl.col("low")) | (pl.col("open") > pl.col("high")) |
        (pl.col("close") < pl.col("low")) | (pl.col("close") > pl.col("high"))
    )
    
    if invalid_geometry.height > 0:
        print(f"FAILED GATE: Found {invalid_geometry.height} rows with invalid OHLC geometry (e.g. low > high).")
        sys.exit(1)
        
    # Gate 3: Negative prices or ranges
    negative_prices = df.filter(
        (pl.col("open") < 0) | (pl.col("high") < 0) | 
        (pl.col("low") < 0) | (pl.col("close") < 0)
    )
    
    if negative_prices.height > 0:
        print(f"FAILED GATE: Found {negative_prices.height} rows with negative prices.")
        sys.exit(1)
        
    # Gate 4: Suspicious zero/stale candles (Volume = 0 or No Price Movement)
    # (Just a warning for now, as illiquid minutes can exist)
    zero_volume = df.filter(pl.col("volume") <= 0)
    if zero_volume.height > 0:
        print(f"WARNING: Found {zero_volume.height} rows with zero or negative volume.")
        
    print("ALL DATA QUALITY GATES PASSED.")

if __name__ == "__main__":
    run()
