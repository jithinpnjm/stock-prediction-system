import polars as pl

def aggregate_1m_to_5m(df_1m: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregates 1-minute OHLCV data into 5-minute candles.
    Ensures that the timestamps align with market session boundaries.
    """
    # Ensure datetime is sorted
    df = df_1m.sort("datetime")
    
    # Standard 5-minute aggregation
    df_5m = df.group_by_dynamic(
        "datetime", 
        every="5m", 
        closed="left", 
        label="left"
    ).agg([
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    ])
    
    return df_5m

def validate_aggregation(df_5m: pl.DataFrame) -> bool:
    """
    Validates the 5m aggregated dataframe.
    """
    if df_5m.is_empty():
        return False
        
    # Check for negative spreads
    invalid_spreads = df_5m.filter(
        (pl.col("high") < pl.col("low")) | 
        (pl.col("high") < pl.col("open")) |
        (pl.col("high") < pl.col("close")) |
        (pl.col("low") > pl.col("open")) |
        (pl.col("low") > pl.col("close"))
    )
    
    if len(invalid_spreads) > 0:
        raise ValueError(f"Found {len(invalid_spreads)} candles with invalid OHLC geometry.")
        
    return True
