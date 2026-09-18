import polars as pl
import os
import numpy as np

def generate_mock_data():
    """Generates synthetic 1m data for local pipeline testing."""
    os.makedirs("data/raw", exist_ok=True)
    
    from datetime import datetime, timedelta
    base = datetime(2023, 1, 1, 9, 15)
    
    # Generate 5,000 minutes of data (~13 days of Bank Nifty)
    dates = [base + timedelta(minutes=i) for i in range(5000)]
    
    # Simulate a random walk for prices
    np.random.seed(42)
    returns = np.random.normal(0, 5, size=5000)
    close = 40000 + np.cumsum(returns)
    
    open_prices = close - np.random.normal(0, 2, size=5000)
    high_prices = np.maximum(open_prices, close) + np.random.uniform(0, 10, size=5000)
    low_prices = np.minimum(open_prices, close) - np.random.uniform(0, 10, size=5000)
    
    df = pl.DataFrame({
        "datetime": dates,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close,
        "volume": np.random.randint(100, 5000, size=5000)
    })
    
    df.write_parquet("data/raw/1m_data.parquet")
    print(f"Generated mock data at data/raw/1m_data.parquet with {df.height} rows.")

if __name__ == "__main__":
    generate_mock_data()
