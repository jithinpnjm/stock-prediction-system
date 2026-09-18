# Legacy / Canonical Module Reconciliation

This document prevents future agents from treating similarly named inherited modules as separate competing architectures.

## Canonical modules used by the current pipeline

| Concern | Canonical path | Notes |
|---|---|---|
| Candle geometry | `src/features/candles.py` | Base OHLC geometry and session-local returns |
| Candle clusters | `src/features/clusters.py` | 1-10 bar structural clusters |
| Volatility | `src/features/volatility.py` | Session-local ATR/volatility |
| Swings | `src/features/swings.py` | Causal, session-local swing candidates |
| Market structure | `src/features/market_structure.py` | Swing state, HH/HL/LH/LL, BOS-style context |
| Higher timeframe | `src/features/multitimeframe.py` | Canonical MTF implementation used by pipeline |
| Volume/VWAP | `src/features/session_vwap.py` plus cluster features | VWAP is opt-in; legacy volume features are retained only for older experiments |
| Event sampling | `src/features/event_sampling.py` | Session-local CUSUM |
| Sequence construction | `src/models/sequence_data.py` | Session-boundary and 5-minute-contiguity enforcement |
| TCN | `src/models/tcn.py` | Canonical TCN |
| Transformer | `src/models/transformer.py` | Canonical Transformer |
| Validation split | `src/validation/purged_cv.py` | Canonical chronological purged split |
| CPCV | `src/validation/cpcv.py` | Canonical interval-aware CPCV utility |
| Inference decision | `src/inference/decision.py` | Canonical probability/edge/NO_TRADE policy |
| Backtest | `src/backtest/engine.py` | Canonical event-driven engine |

## Inherited modules still present

### `src/features/multi_timeframe.py`

Older MTF implementation. The pipeline imports `src.features.multitimeframe`. Do not create new features here. Before deleting it, search historical experiments and tests and migrate anything still needed.

### `src/features/structure_state.py`

Older simple structure proxy. The canonical market-structure engine is `market_structure.py`. Retain this file only until its independent uses are reviewed.

### `src/features/volume.py`

Older volume/VWAP feature set. It is not part of the canonical feature pipeline. It has been made session-local so older experiments cannot accidentally cross the overnight boundary.

### `src/models/torch_models.py`

Older standalone TCN/Transformer implementation. Canonical models live in `tcn.py` and `transformer.py`. It should not receive new model development.

### `src/models/sequences.py`

Older sequence builder. Canonical sequence construction is `sequence_data.py`, which now enforces in-session 5-minute continuity.

### `src/backtest/decision.py`

Older decision adapter. New decision logic belongs in `src/inference/decision.py`. Migrate references before deleting the adapter.

### `src/common/contracts.py`

Shared market constants and dataclasses inherited from the earlier implementation. Review before replacing because some legacy tests/configs may still reference it.

## Cleanup rule

Do not delete an inherited module merely because a canonical replacement exists. First:

1. Search repository references.
2. Compare behavior against the canonical implementation.
3. Migrate tests and pipeline references.
4. Remove only when no supported workflow depends on it.
5. Record the removal in Git history and `docs/project-status.md`.

The objective is one canonical implementation per concern, not a compatibility maze.
