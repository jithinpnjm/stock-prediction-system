import polars as pl

from src.backtest.metrics import calculate_metrics


def run():
    print("Running pipeline step: 10_report.py")

    try:
        equity_curve = pl.read_parquet("data/backtest/equity_curve.parquet")
        trades_df = pl.read_parquet("data/backtest/trades.parquet")
    except FileNotFoundError:
        print("Run 09_backtest.py first.")
        return

    # To calculate metrics properly, we map the polars dataframe back to our Trade objects
    from src.backtest.portfolio import Trade

    trades = [
        Trade(
            entry_time=row["entry_time"],
            entry_price=0,
            direction=row["direction"],
            size=0,
            exit_time=row["exit_time"],
            exit_price=0,
            pnl=row["pnl"],
        )
        for row in trades_df.to_dicts()
    ]

    metrics = calculate_metrics(equity_curve, trades)

    print("\n" + "=" * 40)
    print("   BACKTEST PERFORMANCE REPORT")
    print("=" * 40)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:<20}: {v:.2f}")
        else:
            print(f"{k:<20}: {v}")
    print("=" * 40)


if __name__ == "__main__":
    run()
