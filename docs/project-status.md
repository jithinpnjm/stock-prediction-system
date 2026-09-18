# Project Status — Bank Nifty Research Platform

Updated: 2026-09-18

## Purpose

Build a research-grade system that determines whether completed Bank Nifty SPOT 5-minute price action contains robust out-of-sample information about reaching favorable intraday price objectives. The intended minimum favorable target is 200 points, with a 70-point adverse barrier and the ability to research larger dynamic targets.

The goal is not maximum historical backtest profit. The goal is a reproducible, leakage-safe, statistically defensible research process that can later be executed on Nebius L40S GPU nodes.

## Current Git state

- Main research foundation is merged into `main`.
- Earlier PR #2 was merged, then reverted after merge-history/conflict handling.
- Fresh PR #3 re-applied the research foundation against the reverted `main` state and was merged.
- Current `main` at the start of this cleanup phase: `688ed0917576e0d3a35f122066b7df3031f6942f`.
- Current cleanup branch: `cleanup-and-continuity`.
- Do not treat the PR/merge history as the research result; the repository contents and Git history are the durable implementation record.

## Important historical/data decisions

### Timestamp contract

FYERS 1-minute timestamps are treated as candle START times.

Canonical data therefore uses:
- `source_timestamp`: original source timestamp.
- `timestamp`: source timestamp shifted by one minute to represent availability/close semantics.

For 5-minute aggregation, a completed bar is represented by its final 1-minute availability timestamp, for example 09:20, 09:25, ..., 15:30 IST.

This distinction is mandatory for leakage and execution alignment.

### Dataset policy

Raw 1-minute data is immutable and remains the source of truth.

Legacy committed market-data snapshots in `data/` were preserved during the rebuild. DVC migration is documented but must not be considered complete until a remote is configured and a clean-clone restore is verified.

Real Fyers credentials are local-only and must never be committed.

## What is implemented

### Data
- Canonical schema and timestamp contract.
- NSE calendar/session handling.
- Source ingestion and immutable snapshot helpers.
- FYERS history client with chunking/retry/rate-limit handling.
- 1-minute validation gates.
- 1m -> 5m aggregation with complete-bucket validation.

### Features
- Candle geometry and returns.
- Consecutive candle cluster descriptors for 1-10 bars.
- Session time features.
- ATR/volatility features.
- Opening-range features.
- Session context and prior-day context.
- Session-local support/resistance.
- Causal swing candidates.
- Market-structure state and BOS-style features.
- Causal multi-timeframe context.
- Historical same-time-slot context from prior days.
- Optional session VWAP.
- CUSUM/event-driven features.
- Optional fractional differentiation.
- 1-minute-inside-5-minute microstructure.
- Research hooks for analogue search, embeddings, pattern mining, feature selection, and regime clustering/HMM.

### Labels
- Path-aware triple-barrier labels.
- Configurable 200/250/300/350/400/500 target ladder.
- MFE/MAE metadata.
- Barrier timing.
- Conservative same-1m-bar target/stop collision handling.

### Models
- Logistic baseline.
- LightGBM.
- XGBoost.
- TCN.
- Transformer.
- Meta-label model.
- MFE/magnitude model.
- Survival/hazard benchmark.
- Probability ensemble and abstention logic.
- Dynamic target selector.

### Validation
- Purged chronological split.
- Embargo helpers.
- CPCV utilities.
- Calibration utilities.
- Bootstrap/statistical diagnostics.
- Robustness/PBO-style diagnostics.
- Frozen chronological holdout.
- Leakage assertions.

### Backtesting
- Event-driven stateful backtester.
- 1-minute execution/path.
- Latency.
- Slippage and commission.
- Conservative same-bar ambiguity handling.
- Session flatten.
- PnL/Sharpe/Drawdown-style metrics.

### MLOps
- Git/dataset/feature/label/validation lineage.
- SHA-256 manifests.
- Reproducibility snapshot.
- MLflow helpers.
- Model registry helpers.
- Experiment-budget guard.

### Pipelines
The repository contains ingest -> validate -> aggregate -> feature -> label -> dataset -> train -> validate -> backtest -> report -> registry stages, plus sequence and sweep entry points.

## What is NOT completed yet

These are implementation/research gaps, in priority order.

### P0 — software foundation
1. Make the full test suite pass on the merged `main`.
2. Reconcile inherited legacy APIs/modules with the canonical rebuild.
3. Remove or explicitly mark duplicate implementations so there is one canonical path.
4. Finish session-boundary/cross-day causality audit across every feature.
5. Add deterministic smoke tests for the full pipeline on tiny synthetic data.
6. Ensure optional ML/GPU tests are separated from cheap core CI.

### P1 — reproducible data platform
1. Configure/verify DVC remote.
2. Migrate the legacy market data to versioned DVC artifacts where appropriate.
3. Verify restore from a clean clone.
4. Run FYERS ingestion against local credentials and establish the canonical historical dataset manifest.
5. Record exact source coverage and missing/partial trading days.
6. Never mutate raw snapshots.

### P1 — baseline research
After software CI is green:
1. Build the complete 1m -> 5m -> features -> labels -> training dataset.
2. Run purged walk-forward LightGBM.
3. Save OOF predictions with fold IDs and lineage.
4. Calibrate OOF probabilities.
5. Run the 1-minute-path backtest.
6. Generate the first statistical research report.

No deep-learning model should be treated as the primary candidate until this baseline is measured.

### P1 — model research
1. TCN vs Transformer under identical leakage-safe folds.
2. Regime features and conditional performance.
3. Meta-labeling / trade-quality model.
4. MFE/magnitude model and dynamic target policy.
5. Survival/hazard benchmark.
6. Analogue search and embedding/pattern research as secondary signals.
7. Controlled ensemble/blending only after component OOS behavior is known.

### P1 — statistical audit
For the experiment program:
- track every trial and seed;
- compare best/median/worst trials;
- use bootstrap/permutation diagnostics;
- evaluate subperiod/year/regime behavior;
- evaluate PBO/DSR-style diagnostics;
- stress realistic costs/slippage;
- keep the final holdout untouched.

### P2 — productionization
Only after robust OOS evidence:
- packaged inference path;
- model registry promotion policy;
- live 1m -> completed 5m feature path;
- prediction logging;
- data/model/feature drift monitoring;
- emergency disable;
- retraining procedure;
- operational alerts.

## Nebius execution contract

The intended heavy-compute environment is a Nebius L40S GPU node.

When GPU execution starts later, the run should record:

```
git commit
dataset_id
dataset manifest SHA
feature_version
label_version
validation_version
model/config version
random seeds
training command
hardware/device
MLflow run ID
```

GPU runs should be manually/explicitly launched rather than silently triggered by ordinary CI.

## Recommended build order

```
CI + API reconciliation
        ->
feature/session causality audit
        ->
tiny end-to-end deterministic smoke pipeline
        ->
DVC + clean-clone reproducibility
        ->
real historical ingestion
        ->
LightGBM OOF baseline
        ->
calibration + realistic backtest
        ->
MFE/dynamic target
        ->
TCN / Transformer
        ->
regime + meta-label + secondary research
        ->
CPCV/PBO/DSR audit
        ->
frozen holdout
        ->
live inference/production
```

## Agent handoff rule

When another coding agent takes over, it should begin by reading this file and `AGENTS.md`, inspect `git log` and current CI status, then continue from the first unchecked priority item. It must not restart the architecture from scratch or infer the plan only from scattered code.

The conversation is not the canonical specification. This repository is.
