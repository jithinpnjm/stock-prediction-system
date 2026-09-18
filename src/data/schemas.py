from __future__ import annotations

from dataclasses import dataclass
import polars as pl

REQUIRED_COLUMNS=("timestamp","open","high","low","close","volume")
CANONICAL_NUMERIC=("open","high","low","close","volume")

@dataclass(frozen=True)
class DataContract:
    symbol:str="NSE:NIFTYBANK-INDEX"
    source_timeframe:str="1m"
    model_timeframe:str="5m"
    timezone:str="Asia/Kolkata"
    source_timestamp_semantics:str="start"
    session_open:str="09:15"
    session_close:str="15:30"
    evaluation_open:str="09:30"
    evaluation_close:str="15:00"

def validate_required_schema(df:pl.DataFrame)->None:
    missing=sorted(set(REQUIRED_COLUMNS)-set(df.columns))
    if missing: raise ValueError(f"Missing required columns: {missing}")
    for c in CANONICAL_NUMERIC:
        if not df[c].dtype.is_numeric(): raise TypeError(f"{c} must be numeric")

def _localize(expr:pl.Expr,timezone:str)->pl.Expr:
    return (
        expr.cast(pl.Datetime(time_zone=None),strict=False)
        .dt.replace_time_zone(timezone)
    )

def coerce_canonical_schema(
    df:pl.DataFrame,
    *,
    source_timestamp_semantics:str="start",
    timezone:str="Asia/Kolkata",
)->pl.DataFrame:
    validate_required_schema(df)
    source=_localize(pl.col("timestamp"),timezone)
    if source_timestamp_semantics=="start":
        close=source+pl.duration(minutes=1)
    elif source_timestamp_semantics=="close":
        close=source
    else:
        raise ValueError("source_timestamp_semantics must be start or close")
    return (
        df.with_columns(
            source.alias("source_timestamp"),
            close.alias("timestamp"),
            *[pl.col(c).cast(pl.Float64,strict=False).alias(c) for c in CANONICAL_NUMERIC],
        )
        .sort("timestamp")
    )
