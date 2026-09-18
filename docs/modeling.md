# Modeling

The model stack is layered.

## Baselines

1. Logistic regression sanity check.
2. LightGBM primary tabular benchmark.
3. XGBoost benchmark.

## Sequence models

TCN and Transformer encoders operate on fixed historical 5-minute feature windows.

## Secondary models

Regime clustering/HMM, analogue search, meta-labeling, MFE quantile regression and discrete hazard benchmarks address different research questions.

No deep model is promoted solely because it produces a better historical score. It must demonstrate incremental OOS information relative to the simpler baseline.
