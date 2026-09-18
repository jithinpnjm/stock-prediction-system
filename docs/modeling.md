# Modeling

## Baseline-first policy

The first predictive baseline is LightGBM with a logistic regression reference and
XGBoost/CatBoost as additional tabular models.

The model target is the path-aware event label:

- -1: clean short target reached first
- 0: no clean directional target, stop/time/ambiguous outcome
- +1: clean long target reached first

OOF probabilities are persisted with timestamps. The decision layer may return
NO_TRADE rather than forcing the maximum-probability class.

## Sequence models

TCN and causal Transformer implementations consume rolling 5m feature sequences.
The TCN uses left-only causal padding. The Transformer uses a causal attention mask.

Sequence models should be compared against the tabular baselines under the same
walk-forward protocol before adding complexity.

## Probability quality

Use log loss, balanced accuracy, calibration error and probability stability.
Accuracy by itself is not a trading objective.

## Magnitude and time-to-event

MFE/MAE regressors and time-to-event models are auxiliary research tracks. They
must never leak their future outcomes into the primary feature set.
