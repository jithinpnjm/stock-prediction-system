# External market context

The architecture can add external context without weakening point-in-time
guarantees.

Supported research inputs include:

- Nifty and Sensex index state
- India VIX
- Bank Nifty futures basis and open interest
- historical option-chain features
- constituent breadth and dispersion
- timestamped news or sentiment signals

## Availability semantics

Every external observation must carry an **availability timestamp** representing
when the value became knowable to the trading system.

A backward as-of join may then attach the latest observation whose availability
timestamp is at or before the 5-minute model timestamp.

The period being described is not enough by itself. A published macro value can
describe an earlier period while only becoming available later.

## Historical integrity

External datasets are enabled only after their:

- publication delay,
- timestamp semantics,
- revision behavior,
- survivorship behavior,
- missing-data policy,
- contract/expiry rules,
- and vendor corrections

have been documented and tested.

The initial Bank Nifty system therefore remains reproducible from SPOT 1-minute
data alone. External context is an isolated research extension, not a hidden
dependency of the baseline.
