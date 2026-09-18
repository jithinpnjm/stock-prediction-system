# Labeling

Primary event labels use Bank Nifty SPOT.

## Baseline

- Long: +200 points before -70 points.
- Short: -200 points before +70 points.
- Otherwise: no-event/timeout.
- Horizon: bounded by the configured horizon and same-session data.

The labeler evaluates subsequent 1-minute bars after a completed 5-minute event. This preserves the underlying path needed to resolve what can and cannot be inferred from a 5-minute OHLC bar.

Same-bar target/stop collisions are treated conservatively as adverse-first because tick ordering is not available.

Every sample retains event_end_timestamp and barrier_timestamp for leakage-safe interval purging and audit.
