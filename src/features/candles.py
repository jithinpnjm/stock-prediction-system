import polars as pl

def add_candle_geometry_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds point-in-time structural candle features.
    """
    # 1. Base differences & Wicks
    df = df.with_columns([
        (pl.col("high") - pl.col("low")).alias("f_range"),
        (pl.col("close") - pl.col("open")).alias("f_body"),
        pl.max_horizontal("open", "close").alias("top_body"),
        pl.min_horizontal("open", "close").alias("bottom_body")
    ])
    
    # 2. Wick lengths & Normalized metrics
    df = df.with_columns([
        (pl.col("high") - pl.col("top_body")).alias("f_upper_wick"),
        (pl.col("bottom_body") - pl.col("low")).alias("f_lower_wick"),
    ]).with_columns([
        (pl.col("f_body").abs() / pl.col("f_range").fill_null(1.0).fill_nan(1.0)).alias("f_body_pct"),
        (pl.col("f_upper_wick") / pl.col("f_range").fill_null(1.0).fill_nan(1.0)).alias("f_upper_wick_pct"),
        (pl.col("f_lower_wick") / pl.col("f_range").fill_null(1.0).fill_nan(1.0)).alias("f_lower_wick_pct"),
        
        # Direction
        pl.when(pl.col("close") > pl.col("open")).then(pl.lit(1))
          .when(pl.col("close") < pl.col("open")).then(pl.lit(-1))
          .otherwise(pl.lit(0)).alias("f_direction")
    ])
    
    return df.drop(["top_body", "bottom_body"])
