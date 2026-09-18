import polars as pl
import joblib
import os
from src.models.lightgbm import predict_lightgbm
from src.backtest.engine import EventDrivenBacktester

def run():
    print("Running pipeline step: 09_backtest.py")
    
    try:
        # Load canonical pricing data to get datetimes and close prices
        df_prices = pl.read_parquet("data/silver/5m_canonical.parquet")
        
        # Load features for inference
        X = pl.read_parquet("data/ml/X.parquet").to_numpy()
        
        # Load trained model
        models = joblib.load("models/lightgbm_cv_models.pkl")
        model = models[0] # Using fold 1 for prototype demo
    except FileNotFoundError:
        print("Run prior pipelines up to 08_validate.py first.")
        return

    # Generate Predictions
    predictions = predict_lightgbm(model, X)
    
    # Merge predictions into the price dataframe
    df_backtest = df_prices.with_columns(
        pl.Series("signal", predictions)
    )
    
    # Run Backtest
    backtester = EventDrivenBacktester(df_backtest, initial_capital=100000.0)
    equity_curve, trades = backtester.run()
    
    # Save artifacts
    os.makedirs("data/backtest", exist_ok=True)
    equity_curve.write_parquet("data/backtest/equity_curve.parquet")
    
    # Save trades
    trades_dict = [
        {"entry_time": t.entry_time, "exit_time": t.exit_time, "direction": t.direction, "pnl": t.pnl} 
        for t in trades
    ]
    pl.DataFrame(trades_dict).write_parquet("data/backtest/trades.parquet")
    
    print(f"Backtest completed. Extracted {len(trades)} trades.")

if __name__ == "__main__":
    run()
