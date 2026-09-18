# Legacy / Canonical Module Reconciliation

This document prevents future agents from treating similarly named inherited modules as separate competing architectures.

## Canonical modules

| Concern | Canonical path | Notes |
|---|---|---|
| Feature orchestration | `src/features/engine.py` | Single entry point used by `pipelines/04_features.py` |
| Candle geometry | `src/features/candles.py` | Base OHLC geometry and returns |
| Candle clusters | `src/features/clusters.py` | 1-10 bar structural clusters |
| Volatility | `src/features/volatility.py` | Session-local ATR/volatility |
| Swings | `src/features/swings.py` | Causal, session-local swing candidates |
| Market structure | `src/features/market_structure.py` | Swing state, HH/HL/LH/LL, BOS-style context |
| Higher timeframe | `src/features/multitimeframe.py` | Canonical MTF implementation |
| Session context | `src/features/session_context.py` | Prior-day and session-to-date context |
| Opening range | `src/features/opening_range.py` | Session-local opening windows |
| Support/resistance | `src/features/support_resistance.py` | Session-local structure |
| Historical intraday | `src/features/historical_intraday.py` | Prior-day same-slot context |
| Volume/VWAP | `src/features/session_vwap.py` plus cluster features | VWAP is opt-in |
| Event sampling | `src/features/event_sampling.py` | Session-local CUSUM |
| Microstructure | `src/features/microstructure.py` | 1m path inside completed 5m event |
| Fractional differentiation | `src/features/fractional_diff.py` | Opt-in research branch |
| Sequence construction | `src/models/sequence_data.py` | Session-boundary and 5m-contiguity enforcement |
| TCN | `src/models/tcn.py` | Canonical TCN |
| Transformer | `src/models/transformer.py` | Canonical Transformer |
| Validation split | `src/validation/purged_cv.py` | Chronological purged split |
| CPCV | `src/validation/cpcv.py` | Interval-aware CPCV |
| Inference decision | `src/inference/decision.py` | Probability/edge/NO_TRADE policy |
| Backtest | `src/backtest/engine.py` | Stateful event-driven backtest |

## Removed obsolete duplicates

The following inherited implementations were deleted during cleanup because current supported workflows had canonical replacements and no active references:

- `src/features/multi_timeframe.py`
- `src/features/structure_state.py`
- `src/models/torch_models.py`
- `src/models/sequences.py`
- `src/backtest/decision.py`

The repository history retains the deleted implementations, so future agents can inspect them when needed without reintroducing them into the active architecture.

## Retained legacy support

`src/features/volume.py` remains only for older experiments. It is session-local and is not part of the canonical feature engine.

`src/common/contracts.py` remains shared by some inherited tests/configuration code and should be removed only after a dedicated reference audit and migration.

## Cleanup rule

Do not create a second implementation for a concern that already has a canonical path. Before adding or reviving legacy code:

1. Search repository references.
2. Compare the required behavior with the canonical implementation.
3. Update the canonical implementation or migrate the reference.
4. Add a regression test.
5. Document the decision in `docs/project-status.md`.

The objective is one canonical implementation per concern, not a compatibility maze.
