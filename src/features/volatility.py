import polars as pl

def add_volatility_features(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """
    Adds Average True Range (ATR) and Normalized Volatility features.
    """
    # Calculate True Range (TR)
    # TR = max(High - Low, abs(High - PrevClose), abs(Low - PrevClose))
    
    df = df.with_columns(
        pl.col("close").shift(1).alias("prev_close")
    ).with_columns([
        (pl.col("high") - pl.col("low")).alias("tr_1"),
        (pl.col("high") - pl.col("prev_close")).abs().alias("tr_2"),
        (pl.col("low") - pl.col("prev_close")).abs().alias("tr_3"),
    ]).with_columns(
        pl.max_horizontal("tr_1", "tr_2", "tr_3").alias("true_range")
    )
    
    # Simple moving average of TR for ATR
    df = df.with_columns(
        pl.col("true_range").rolling_mean(window_size=period).alias(f"f_atr_{period}")
    ).with_columns([
        # Normalized ATR
        (pl.col(f"f_atr_{period}") / pl.col("close") * 10000).alias(f"f_natr_{period}_bps"),
        
        # Current Range relative to ATR
        (pl.col("tr_1") / pl.col(f"f_atr_{period}").fill_null(1.0).fill_nan(1.0)).alias("f_range_to_atr")
    ])
    
    # Cleanup intermediate columns
    return df.drop(["prev_close", "tr_1", "tr_2", "tr_3", "true_range"])
