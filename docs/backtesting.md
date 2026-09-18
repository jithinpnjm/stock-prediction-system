# Backtesting

Backtests consume model predictions generated at completed 5-minute timestamps and execute against subsequent 1-minute source bars.

The execution model is explicit about:

- signal timestamp;
- latency;
- entry/exit slippage;
- fixed order commission;
- target/stop path;
- ambiguous same-bar collisions;
- evaluation entry window;
- session flattening.

The current engine is a research benchmark, not a broker emulator. Broker-specific charges, spreads, impact and instrument-specific execution must be added before any production/live-trading conclusion.
