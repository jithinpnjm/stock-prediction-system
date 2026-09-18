# Feature Catalog

## Price action
Candle range, body, wick proportions, candle direction, close location and
normalized bps movement.

## Clusters
Consecutive same-direction runs from length 1 through 10, cumulative body/range,
volume-relative measures and wick statistics.

## Time/session
Minutes from open/close, cyclical time-of-day, opening/closing windows, weekday.

## Volatility
ATR(5/14/30), range-to-ATR, normalized ATR, multi-period returns, realized
volatility and rolling range.

## Opening structure
Prior opening ranges, location inside the opening range and breakout flags.

## Market structure
Causal swing candidates, prior swing extremes, breakout candidates and normalized
distance to structure.

## Support/resistance
Causal rolling extrema and ATR-normalized distance.

## Multi-timeframe
15m/30m/60m context derived only from already completed 5m bars.

## Microstructure
1m-inside-5m minute count, directional minute ratio, intrabar path efficiency,
realized minute variability, path movement and range.

## Event sampling
CUSUM events for research-oriented event-based sampling.

All features must remain point-in-time and traceable to source timeframe and lookback.
