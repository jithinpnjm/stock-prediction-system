# Validation

Temporal validation is mandatory.

## Rules

- No random train/test split.
- Test blocks are chronological.
- Training events whose outcome interval overlaps the test period are purged.
- An embargo is applied after the test start.
- Hyperparameters are selected without touching the final holdout.
- OOF predictions, not in-sample predictions, are used for performance evidence.

CPCV and robustness utilities provide additional research diagnostics. The final holdout must remain frozen until the research program is explicitly locked.
