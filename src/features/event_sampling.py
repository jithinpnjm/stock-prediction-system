from __future__ import annotations

import polars as pl


def add_cusum_events(df: pl.DataFrame, threshold_atr: float = 1.0) -> pl.DataFrame:
    # CUSUM state is deliberately implemented in Python so the event detector
    # is explicit and testable. It only consumes information up to each row.
    returns = df.select(
        [
            "timestamp",
            (pl.col("close").log().diff().fill_null(0.0)).alias("log_return"),
            pl.col("atr_14").fill_null(0.0).alias("atr"),
            pl.col("close").alias("close"),
        ]
    ).to_dicts()

    pos = 0.0
    neg = 0.0
    events: list[dict[str, object]] = []
    for row in returns:
        scale = max(float(row["atr"]) / max(abs(float(row["close"])), 1e-9), 1e-6)
        ret = float(row["log_return"])
        pos = max(0.0, pos + ret)
        neg = min(0.0, neg + ret)
        triggered = pos > threshold_atr * scale or abs(neg) > threshold_atr * scale
        events.append({"timestamp": row["timestamp"], "cusum_event": int(triggered)})
        if triggered:
            pos = 0.0
            neg = 0.0

    return df.join(pl.DataFrame(events), on="timestamp", how="left")
