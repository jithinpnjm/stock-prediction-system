# Bank Nifty Trading Research Plan

Updated: 2026-09-18

## Objective

This is not a generic ML classifier or a GPU optimization exercise. The research question is whether completed Bank Nifty SPOT 5-minute price-action configurations contain stable, reproducible, out-of-sample information that can identify sufficiently large intraday opportunities after realistic costs. The system must be selective and may return NO_TRADE.

Initial economic objective: LONG +200 before -70, or SHORT -200 before +70, using the 1-minute future path for labels and execution. Targets 250/300/350/400/500+ are research candidates, not assumptions.

## 1. Primary research unit: 5-minute candle clusters

Research completed 5-minute clusters of lengths 1 through 10. A cluster is not merely a sequence of green/red candles. Represent:
- direction sequence and directional balance
- consecutive up/down structure
- cumulative/net return
- accumulated range and body
- range efficiency: net movement / accumulated range
- wick and close-location geometry
- cumulative/average/relative volume
- ATR-relative range
- compression/expansion state.

Do not hard-code rules such as three green candles = BUY. Discover cluster/context combinations and validate them out of sample.

## 2. Context around each cluster

### VWAP

Session VWAP is a core contextual feature family: price-to-VWAP distance, ATR-normalized distance, VWAP slope, above/below state, crosses, reclaims, rejections, and persistence. Identical clusters in different VWAP states are different candidate market states. VWAP is context, not a predetermined rule.

### Market structure

Use causal swing highs/lows, HH/HL/LH/LL, structural trend, breakout/breakdown, structure breaks, range, compression, and breakout/retest context. Future-confirmed swings must never enter a prediction timestamp.

### Support/resistance

Use previous-day high/low/close, intraday and swing levels, rolling support/resistance, distance to levels, breakout distance, and proximity to major levels. Target feasibility depends on available room to structural barriers.

### Volatility

Use ATR 6/14/30, ATR ratios, range/ATR, true-range statistics, volatility shock, compression, expansion, recent contraction, and relative expansion. Focus on transitions such as compression -> volume expansion -> range expansion -> directional cluster, not simply high volatility.

### Volume

Use raw/rolling/relative volume, spikes, acceleration, recent/session baseline, and directional candle-volume relationships.

### Session/time

Use minutes from open/close, session bar index/phase, opening-range state, first 30/60-minute context, midday/afternoon/late-session behavior. Test time-of-day effects rather than assuming them.

### Opening range

Research first 3 and first 6 completed 5-minute bars: range high/low/width, position, distance, breakout/breakdown, post-breakout expansion, and rejection back into range.

### Previous-day context

Use previous-day return, range, high, low, close, and gap from previous close. Only information known before the current prediction is allowed.

### Historical same-time-slot context

Use prior trading days at the same intraday slot: mean return, return volatility, and positive-rate over a configurable lookback. The current day's future must never enter the history.

### 1-minute microstructure

Within each completed 5-minute bar, use available 1-minute path information: path high/low/range, net movement, internal range statistics, directional consistency, up/down ratio, volume, rejection, and efficiency. Raw 1-minute data remains the source of truth.

## 3. Labels

LONG: +200 before -70. SHORT: -200 before +70. Otherwise timeout/session exit. Barrier ordering comes from the 1-minute path. Preserve direction, outcome, target, stop, barrier time, time-to-event, MFE, and MAE. Same-bar ambiguity uses the repository's conservative rule. Future label/path data must never become features.

## 4. Do not collapse the problem into one model

Research separate questions:

1. Direction: LONG versus SHORT.
2. Trade quality/meta-label: is this directional signal worth taking?
3. Magnitude: does the setup support 200/250/300/350/400/500+?
4. Path/timing: probability of target-first, time to target, MFE, MAE and adverse excursion.

The final decision combines direction, trade quality, magnitude/path evidence, regime/context, calibration, and realistic risk/cost checks. It may output LONG, SHORT, or NO_TRADE, with a selected target.

High directional confidence alone must never justify a larger target.

## 5. Feature-selection funnel before expensive GPU work

Evaluate incremental value using identical purged chronological folds:

cluster only -> +VWAP -> +volatility/volume -> +structure -> +support/resistance -> +session/opening range -> +historical intraday -> +1m microstructure.

Remove or demote feature families that do not provide stable incremental OOS information. Do not carry every available feature into a large Transformer by default.

## 6. Cheap discovery before GPU

For important cluster/context families measure frequency, LONG/SHORT distribution, target/stop/timeout rates, MFE/MAE distributions, and stability by regime, time of day, volatility, VWAP state, structure and calendar subperiod. The purpose is to determine whether enough signal exists to justify expensive representation learning.

## 7. First serious model: LightGBM

Use LightGBM as the first nonlinear baseline with purged walk-forward validation, embargo, identical folds, OOF predictions, fold IDs, reproducible seeds, calibration, and realistic 1-minute-path backtesting. Evaluate probability quality, target outcomes, trade-quality filtering, OOS stability, costs/slippage, and subperiod/regime robustness.

If the simpler baseline cannot demonstrate stable OOS information, do not assume a deeper model will rescue it.

## 8. OOF and calibration

Canonical loop: training fold -> validation fold -> OOF prediction -> calibration -> threshold/target research -> 1-minute-path backtest. Never use the frozen holdout for feature selection, hyperparameter tuning, threshold/target selection, calibration decisions, or ensemble weighting.

## 9. Dynamic target research

Evaluate 200/250/300/350/400/500 using target-first probability, stop-first probability, MFE, MAE, time-to-target, cost-adjusted expectancy and OOS stability. Larger targets require OOS magnitude/MFE/path evidence.

## 10. Regime research

Evaluate performance conditional on trend/range, compression/expansion, volatility, VWAP state, opening-range state, structure and time of day. Optional clustering/HMM regimes must be fitted only on training data inside each validation procedure.

## 11. GPU gate

Nebius L40S compute is authorized only after data quality/reproducibility, causal features, cluster/context discovery, LightGBM OOF baseline, calibration, and realistic backtesting establish a credible research hypothesis.

Then compare engineered cluster features against TCN and Transformer representations using identical leakage-safe folds. Deep models must demonstrate incremental OOS value; complexity is not evidence.

## 12. Economic success criteria

Accuracy is not the primary objective. Evaluate selectivity/precision, calibration, target-hit probability, expectancy, PnL after costs, drawdown, risk-adjusted performance, trade count, MFE, MAE, target efficiency, time in trade and stability. A high-accuracy model without robust economic value is not successful.

## 13. Robustness

Any promising result must survive chronological subperiods/years, volatility and market regimes, time-of-day buckets, cluster lengths, target sizes, realistic slippage, commissions and latency. Use appropriate bootstrap/permutation and PBO/DSR-style diagnostics. Track the full experiment population to control data-snooping risk.

## 14. Eventual prediction system

Live architecture later: live 1m -> completed 5m -> current cluster/context -> calibrated model(s) -> target/risk decision -> LONG/SHORT/NO_TRADE. Historical, backtest, paper-trading and live inference must share the same canonical causal feature contract.

Do not rush live inference before robust OOS evidence and a locked final evaluation.

## 15. Anti-GPU-waste rule

Never do: all features -> huge Transformer -> optimize PnL -> declare success.

Do: 5m cluster discovery -> cheap conditional research -> feature ablation -> LightGBM OOF -> calibration -> trade-quality filtering -> MFE/magnitude -> realistic backtest -> robustness -> TCN/Transformer -> controlled ensemble -> frozen holdout.

Every expensive experiment requires a hypothesis, expected learning, dataset/feature/label/validation versions, seed, MLflow run, compute/runtime record, OOS result and conclusion. Stop experiments that repeatedly fail to demonstrate incremental OOS value.

## 16. Canonical research hierarchy

5m cluster discovery -> context discovery -> causal feature selection -> LightGBM OOF -> calibration -> trade-quality filtering -> MFE/magnitude -> target selection -> realistic 1m-path backtest -> regime analysis -> TCN -> Transformer -> controlled ensemble -> statistical robustness -> frozen holdout -> paper trading -> live inference.

At every step ask: does this add stable out-of-sample information that survives realistic trading assumptions?

## 17. Relationship to engineering documentation

`docs/project-status.md` defines engineering implementation status and infrastructure roadmap.
`docs/trading-research-plan.md` defines the actual trading hypothesis, cluster/context hierarchy, prediction hierarchy, target-selection logic and GPU research gates.
`AGENTS.md` defines continuation and repository integrity rules.

The repository, not conversation history, is the canonical specification.