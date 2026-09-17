#!/usr/bin/env bash
# run_real_sweep.sh
# -------------------
# Phase 3 real sweep, run ON the Nebius VM. All 7 walk-forward folds, real
# epoch counts, every architecture (single-timeframe + the 1min+5min fusion),
# then 2 extra seeds for whichever config comes out ahead so the promotion
# gate's seed-consistency check has something to evaluate.
#
# Everything logs to MLflow (http://localhost:5000) as real nested runs.
# Run this in the background and tail sweep_log.txt for progress.
set -euo pipefail
cd ~/banknifty_bigmove_ml

EPOCHS=15
LOG=sweep_log.txt
: > "$LOG"

run() {
  echo "=== $* ===" | tee -a "$LOG"
  sudo docker run --rm --gpus all --network host \
    -v ~/banknifty_bigmove_ml:/workspace \
    -e MLFLOW_TRACKING_URI=http://localhost:5000 \
    -w /workspace/training \
    banknifty-train:latest \
    "$@" 2>&1 | tee -a "$LOG"
}

echo "### Baselines (all 7 folds) ###" | tee -a "$LOG"
run python3 run_experiment.py --model baseline_naive
run python3 run_experiment.py --model baseline_logreg

echo "### Single-timeframe architectures, seed 0, all 7 folds, $EPOCHS epochs ###" | tee -a "$LOG"
run python3 run_experiment.py --model tcn --epochs $EPOCHS --seed 0
run python3 run_experiment.py --model lstm --epochs $EPOCHS --seed 0
run python3 run_experiment.py --model transformer --epochs $EPOCHS --seed 0

echo "### Multi-timeframe (1min TCN + 5min LSTM), seed 0, all 7 folds ###" | tee -a "$LOG"
run python3 run_experiment_mtf.py --branch-1min tcn --branch-5min lstm --epochs $EPOCHS --seed 0

echo "### Sweep pass 1 complete — see $LOG for per-fold and aggregate metrics ###" | tee -a "$LOG"
echo "Next: pick the winner, rerun it with seed=1 and seed=2, then run promotion_gate.py" | tee -a "$LOG"
