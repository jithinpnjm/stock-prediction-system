import polars as pl
from src.data.aggregate import aggregate_1m_to_5m, validate_aggregation
import os

def run():
    print("Running pipeline step: 03_aggregate_5m.py")
    
    RAW_DATA_PATH = os.environ.get("RAW_1M_DATA_PATH", "data/raw/1m_data.parquet")
    
    try:
        df_1m = pl.read_parquet(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Could not read {RAW_DATA_PATH}. Run 01_ingest.py first.")
        return
        
    df_5m = aggregate_1m_to_5m(df_1m)
    validate_aggregation(df_5m)
    
    os.makedirs("data/silver", exist_ok=True)
    df_5m.write_parquet("data/silver/5m_canonical.parquet")
    print(f"Aggregated data saved. Shape: {df_5m.shape}")

if __name__ == "__main__":
    run()