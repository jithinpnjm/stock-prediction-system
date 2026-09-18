# Architecture

The platform is an event-prediction research system, not a generic next-candle forecaster.

## Data flow

1. Fyers 1-minute Bank Nifty SPOT source.
2. Immutable raw snapshots under data/raw.
3. Validated 1-minute bronze dataset.
4. Session-contained close-timestamped 5-minute canonical dataset.
5. Causal price-action, session, structure and volatility features.
6. 1-minute-path-aware triple-barrier outcomes plus MFE/MAE.
7. Trainable datasets with event intervals.
8. Purged walk-forward models produce OOF predictions.
9. Calibrated/abstaining decisions feed the execution-aware backtester.
10. MLflow records experiment lineage; DVC records data lineage.

All model decisions must be keyed by timestamp, never by row position.
