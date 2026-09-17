# Experiment Policy

## Objective

The project permits large-scale experimentation, including thousands of training runs, but experiment selection must not become test-set optimization.

## Required lineage

Every experiment records:

- Git commit SHA;
- DVC dataset/version identity;
- feature-set version;
- label version;
- validation/split configuration;
- model architecture and hyperparameters;
- random seed;
- software/runtime version;
- training hardware;
- predictions, not only aggregate metrics.

## Validation hierarchy

```text
Development / training folds
        -> walk-forward OOS validation
        -> robustness and regime analysis
        -> frozen holdout
        -> paper/shadow trading
        -> production candidate
```

The frozen holdout is not used for feature selection, hyperparameter tuning, threshold selection, or model ranking before the research program is frozen.

## Multiple testing

Large experiment counts increase the probability of discovering spurious historical relationships. Reports must therefore include:

- number of trials searched;
- experiment family;
- fold-by-fold results;
- seed stability;
- regime stability;
- parameter sensitivity;
- performance degradation from development to OOS;
- backtest-overfitting diagnostics where applicable.

Where a strategy is selected after extensive search, statistical claims must account for the search process rather than treating the final trial as an independent hypothesis.

## Promotion principles

No model is promoted solely because it has the highest historical return or accuracy. Promotion requires passing data-quality, leakage, temporal validation, calibration, robustness, execution-cost, and reproducibility gates.

The system must support `NO_TRADE`. A lower trade frequency with stronger validated conditional edge is preferable to forcing a prediction at every timestamp.
