# AGENTS.md — Bank Nifty Research Continuation Guide

This repository is the durable source of truth for the Bank Nifty intraday research project. Future coding agents (Codex, Claude, Gemini, etc.) should use the repository documentation and Git history rather than relying on conversation memory.

## Before changing code

Read, in this order:

1. `README.md` — research objective and architecture.
2. `docs/project-status.md` — current implementation state, decisions, and ordered next work.
3. The relevant design document under `docs/`.
4. Git history for the relevant module before replacing or deleting existing code.

Do not assume that a file existing means the corresponding research task is complete. "Implemented" and "validated on real historical data" are separate states.

## Non-negotiable research rules

- Raw Bank Nifty SPOT 1-minute data is the source of truth.
- FYERS source timestamps are candle start times; the canonical availability timestamp is the source timestamp plus one minute.
- Canonical model data is 5-minute, with the timestamp representing availability at the completed 5-minute close.
- Features must be point-in-time causal: a feature at event time t may use only information available at or before t.
- Session-local rolling/stateful features must reset at NSE session boundaries unless the feature is explicitly historical cross-day context.
- Triple-barrier labels use the 1-minute path for barrier ordering and never leak future path information into features.
- Chronological, purged, embargoed validation is mandatory for overlapping labels.
- The final holdout is frozen and must not be used for feature selection, hyperparameter tuning, ensemble weighting, or target-policy tuning.
- The model may return NO_TRADE.
- Dynamic targets above 200 points must be justified by an out-of-sample magnitude/MFE model; high directional confidence alone is not sufficient.
- Do not claim a strategy works because of in-sample/backtest-only performance.
- Never commit API credentials, tokens, or local auth files.

## Execution policy

This phase is coding/cleaning/reconciliation only. Do not launch the full five-year research run from GitHub Actions.

Real historical training and GPU-heavy experiments are intended to run on the user's Nebius L40S GPU nodes later. When that phase begins, preserve the exact Git commit, dataset manifest, feature version, label version, validation configuration, and experiment configuration used for every run.

## Persistence policy

Every meaningful implementation batch must be committed and pushed. Do not leave important work only in a transient coding session.

When resolving conflicts, preserve current `main` history and explicitly document the resolution. Do not rewrite shared history unless explicitly requested.
