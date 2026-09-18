# Feature Catalog

## Causal feature families

| Family | Examples |
|---|---|
| Candle geometry | body, range, wick, close location, return |
| Clusters | 1–10 bar return/range/body/direction balance |
| Time | session index, phase, cyclic time, distance to open/close |
| Session | previous-day levels, gap, session high/low |
| Opening range | first 15/30-minute range and position |
| Volatility | ATR, normalized ATR, range shock, realized return |
| Structure | causal breakout candidates, range position, trend context |
| Swings | causal swing candidates and prior-break distances |
| Support/resistance | recent zones, distances, touch diagnostics |
| Event sampling | CUSUM event indicator |

Features derived from future-confirmed information are prohibited. A feature definition should document its source timeframe, lookback and availability timestamp.
