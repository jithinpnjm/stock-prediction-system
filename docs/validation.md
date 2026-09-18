# Validation

Random train/test splits are prohibited.

## Walk-forward OOS

Training folds contain earlier events only. Training events whose information
interval overlaps the test interval are purged. A temporal gap is applied around
the test boundary.

## CPCV

Combinatorial purged cross-validation exists for robustness studies and multiple
path evaluations. It is not used as an excuse to repeatedly tune against one final
holdout.

## Frozen holdout

The last chronologically isolated block is excluded from model selection, threshold
selection and feature selection. Holdout evaluation must be explicitly unlocked.

## Calibration

OOF probabilities are the input to calibration. Accuracy alone is insufficient;
log loss, balanced accuracy, ECE and probability stability are tracked.

## Multiple testing

Every experiment records Git SHA, dataset identity, feature/label versions, split
policy, seed, model parameters, runtime and predictions. Large research sweeps must
be reported as a search family rather than as independent hypotheses.
