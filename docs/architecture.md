# Architecture

## Canonical flow

Fyers 1m source
-> immutable raw session files
-> normalized Bronze 1m
-> validated Bronze
-> canonical Silver 5m
-> point-in-time Gold features
-> path-aware labels
-> training frame
-> OOF model predictions
-> calibrated decision layer
-> execution-aware backtest
-> frozen holdout
-> shadow/paper trading
-> production candidate

## Time semantics

Fyers source timestamps are treated as bar-start timestamps. The 5m layer emits a
close timestamp:

09:15-09:19 -> 09:20
09:20-09:24 -> 09:25
...
15:25-15:29 -> 15:30

All feature availability is defined at the close timestamp. No feature may read
future rows, centered windows, future labels, or post-event information.

## Data ownership

Git stores source code, configuration, schemas, tests and documentation.
DVC stores datasets and reproducible dataset stages.
MLflow stores experiment runs, parameters, metrics and model artifacts.

Raw data is immutable: a new vendor download creates new session files rather than
rewriting historical raw observations.

## Modeling strategy

Baselines are intentionally simple: logistic regression, LightGBM, XGBoost/CatBoost.
Sequence models (TCN and causal Transformer) are research candidates rather than
automatic production replacements.

The decision layer consumes probabilities and may return NO_TRADE. Trade selection
is evaluated on OOF predictions and a separately protected frozen holdout.
