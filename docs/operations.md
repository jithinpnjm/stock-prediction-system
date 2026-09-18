# Operations

## Download

Credentials are loaded from environment variables or local
secrets/fyers_auth.json. The real credentials file is never committed.

Downloaders create immutable per-session files under data/raw/fyers/.

## Daily data checks

1. Verify session count and expected 375 one-minute bars.
2. Check timestamps, duplicates, ordering and session boundaries.
3. Validate OHLC and non-negative volume.
4. Review missing sessions against the maintained holiday calendar.
5. Record a manifest and checksum.

## Research rule

Do not manually patch a raw market-data file. Create a corrected source file or
explicit transformation with a new dataset version.

## Promotion

A candidate requires clean OOS validation, robustness/cost analysis, calibration,
frozen-holdout evaluation and reproducible artifacts before registration.
