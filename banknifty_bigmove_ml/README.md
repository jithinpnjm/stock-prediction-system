# BankNifty "Big Move" ML Prediction System

Full plan: `/Users/jithinpjoseph/.claude/plans/unified-wiggling-wirth.md`

Label (locked): 1 if BankNifty spot touches ±0.3% from a candle's close at
any point within the next 45 one-minute candles, else 0. Computed per
trading session — never leaks across day boundaries.

## Layout
- `data_pipeline/` — Phase 1: label computation (`build_labeled_dataset.py`,
  `label_logic.py` + tests) and walk-forward fold definitions (`splits.py`
  + tests). No infra dependency — runs locally today.
- `infra/` — Phase 0: Nebius VM bootstrap script, MLflow+Postgres
  docker-compose stack, training Dockerfile. See `infra/README.md` for the
  exact setup order once SSH access exists.
- `training/` — Phase 2/3: `windowing.py` (raw-OHLCV sequence windows, tests),
  `models.py` (TCN/LSTM/small-Transformer, tests), `metrics.py`, `baseline.py`
  (naive + logreg reference models), `train_fold.py`, `run_experiment.py`
  (MLflow nested-run orchestrator: parent=config, child=fold), and
  `promotion_gate.py` (the hard gate — never promotes on a single run/fold).
  Fully plumbed and smoke-tested end to end on the Nebius L40S (see below).
- `agent/` — Phase 5, not started.
- `mlflow_config.py` — shared tracking URI/experiment name constants; every
  script should import this rather than hardcoding either.

## Phase 1 — run it now
```
cd banknifty_bigmove_ml/data_pipeline
python3 -m pytest test_label_logic.py test_splits.py -q   # 9 tests, leakage/edge-case checks
python3 build_labeled_dataset.py                            # snapshots raw CSV, labels, writes parquet
python3 splits.py                                            # prints the fold plan for the newest labeled parquet
```

Current real-data result (2026-09-13 run): 463,641 rows kept (ALL of them —
see note below), 407,751 with a label, 55,890 as context-only (session tail)
-> **37.7% positive rate** among labeled rows. Not a rare-event problem —
plain accuracy won't be a misleading metric on its own, but PR-AUC/
calibration still matter since the agent will act on the probability. 7
walk-forward folds + a final untouched holdout (2026-02-16 -> 2026-09-11)
at the default 400-train / 100-val day blocking.

**Important**: `build_labeled_dataset.py` keeps every row, including the
~45/day with `label = NaN`. Those rows are context-only — never a training
target — but are still needed as lookback history for the next day's early
candles (a model predicting on 09:15 needs yesterday's last candles).
Dropping them (an earlier version of this script did) silently breaks
window continuity across every session boundary. Training/windowing code
must filter to `label.notna()` rows for actual targets while still reading
across the full continuous series for input context.

Versioned via DVC (`infra/README.md` step 3) against the Nebius remote —
already round-tripped successfully.

## Phase 2/3 — smoke-tested on the Nebius L40S (2026-09-13)

One fold, 2 epochs, all three reference points logged to MLflow as real
nested runs (`http://localhost:5000` via SSH tunnel):
- `baseline_naive`: ROC-AUC 0.500, PR-AUC 0.228 (= that fold's positive rate, as expected for a constant prediction)
- `baseline_logreg`: ROC-AUC 0.551, PR-AUC 0.298
- `tcn` (2 epochs only): ROC-AUC 0.617, PR-AUC 0.326 — already ahead of both baselines, but this is a 1-fold/2-epoch smoke test, not a real result

`promotion_gate.py` run against the tcn/logreg run IDs correctly refused to
promote: `1/7 folds evaluated` and `no seed-run-ids given` both failed, even
though the single fold it saw did beat baseline. This is the intended
behavior — exactly the discipline that was missing last time.

**Not yet run**: a real multi-fold (all 7), multi-epoch, multi-architecture,
multi-seed sweep — that's a genuine compute-time/cost commitment on the
Nebius VM and should be sized and greenlit explicitly before launching.

## Candle-shape channels + multi-timeframe (2026-09-13)

Two extensions on top of the base pipeline, both still "raw price action,
no hand-picked indicators":

1. **Candle-shape channels** (`windowing.py`, `N_FEATURES=10`): alongside
   raw open/high/low/close-relative-to-anchor, each candle also carries
   body size, upper/lower wick size, range, body-to-range ratio, and color
   (all deterministic functions of OHLC — lossless, just easier for a small
   model to learn from than rediscovering the arithmetic itself).
2. **Multi-timeframe fusion** (`build_multi_timeframe_windows`,
   `MultiTimeframeModel`): a 5-min branch is derived by resampling the SAME
   1-min data (no second data source), using only FULLY CLOSED 5-min bars
   as of each target candle (the in-progress bucket is never visible — this
   is the leakage-safety property `test_htf_branch_never_sees_the_in_progress_bucket`
   in `test_windowing.py` checks directly). Each timeframe gets its own
   encoder branch (TCN/LSTM/Transformer, mix-and-match); pooled outputs are
   concatenated before one shared classification head. `run_experiment_mtf.py`
   is the multi-timeframe counterpart to `run_experiment.py`, same MLflow
   nested-run logging.

Smoke-tested end to end on the L40S (1min-TCN + 5min-LSTM branches, 1 fold,
2 epochs): ROC-AUC 0.597, PR-AUC 0.311 — again, plumbing validation only,
not a real result (2 epochs, 1 fold).

Adding a third timeframe (e.g. 15-min) later is just one more `htf_specs`
entry + one more branch in `MultiTimeframeModel` — no structural change
needed.
