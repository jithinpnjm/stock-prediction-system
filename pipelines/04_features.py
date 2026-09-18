import polars as pl
from src.features.candles import add_candle_geometry_features
from src.features.time_features import add_time_features
from src.features.volatility import add_volatility_features

def run():
    print("Running pipeline step: 04_features.py")
    
    try:
        df_5m = pl.read_parquet("data/silver/5m_canonical.parquet")
    except FileNotFoundError:
        print("Run 03_aggregate_5m.py first.")
        return
        
    # 1. Candle Geometry
    df_features = add_candle_geometry_features(df_5m)
    
    # 2. Time Features
    df_features = add_time_features(df_features)
    
    # 3. Volatility Features (ATR)
    df_features = add_volatility_features(df_features, period=14)
    
    # Drop rows with nulls introduced by rolling features (like ATR)
    df_features = df_features.drop_nulls()
    
    df_features.write_parquet("data/silver/5m_features.parquet")
    print(f"Features engineered. Shape: {df_features.shape}")

if __name__ == "__main__":
    run()
