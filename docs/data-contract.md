# Data Contract

## Scope

This contract defines the canonical Bank Nifty SPOT market-data pipeline. Raw source files are immutable and are never edited in place.

## Time semantics

- Market timezone: `Asia/Kolkata`.
- Every timestamp must be timezone-aware internally.
- A candle timestamp represents the **candle close time** once data reaches the feature/model layer.
- Features at timestamp `t` may use only information observable at or before `t`.
- Operational timezones such as Europe/Berlin must never alter market timestamps.

## Processing layers

```text
Raw 1-minute source
        |
        v
Validated 1-minute canonical data
        |
        v
Canonical 5-minute candles
        |
        v
Point-in-time features
        |
        v
Labels / event outcomes
        |
        v
Training datasets
```

## Required 1-minute fields

The canonical schema must contain, at minimum:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume` (when supplied by the source)

The ingestion layer must retain source-specific columns separately rather than silently discarding information.

## Validation gates

A dataset is trainable only when all critical checks pass:

- timestamps are monotonic within each trading session;
- no duplicate timestamps;
- timestamps belong to valid exchange sessions;
- OHLC relationships are valid: `low <= open`, `low <= close`, `open <= high`, `close <= high`;
- prices and ranges are non-negative;
- unexpected gaps are reported;
- missing candles are classified as expected market closure or data-quality failure;
- schema and dtypes are stable;
- timezone/session conversion is deterministic;
- no implicit forward filling of prices;
- source revisions are detectable where source metadata permits it.

## 5-minute aggregation

Aggregation must be deterministic and versioned. The 1-minute source remains available because 5-minute OHLC alone cannot reliably establish intrabar barrier ordering.

The aggregation implementation must define and test:

- session boundaries;
- 5-minute bucket alignment;
- treatment of incomplete buckets;
- volume aggregation;
- first/last price semantics;
- daylight-saving independence through explicit `Asia/Kolkata` handling.

## Point-in-time feature contract

Every feature should be attributable to:

```text
feature_name
source_columns
source_timeframe
lookback
available_at
transform_version
future_information = false
```

A feature implementation that cannot demonstrate point-in-time availability must not enter the training dataset.

## Versioning

Git versions code/configuration. DVC versions datasets. MLflow versions experiment metadata, metrics, predictions and model artifacts. A training run must record all three lineage identifiers.
