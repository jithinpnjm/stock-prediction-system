# Backtesting

The backtester consumes OOF model predictions joined to market data by timestamp.

Signals never align by row number.

Execution supports configurable:

- latency in bars
- next-bar open or same-bar close execution
- bid/ask spread proxy
- slippage
- flat per-order commission
- variable transaction costs in basis points
- unit sizing and point value

The current implementation is a signal/futures-style point-PnL engine. It is not an
options-pricing engine. Options backtests require contract-level historical option
data, expiry selection, Greeks, spreads, and realistic fill logic.

A research result is incomplete without cost sensitivity, drawdown, trade frequency,
win/loss distribution and stability by regime/time period.
