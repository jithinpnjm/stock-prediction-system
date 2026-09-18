from __future__ import annotations

import polars as pl


def asof_join_context(
    base: pl.DataFrame,
    context: pl.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    prefix: str = "ctx_",
) -> pl.DataFrame:
    """
    Join external observations at or before the model timestamp.

    The context timestamp must represent when the value became available,
    not merely the period it describes. This distinction prevents delayed
    releases from leaking into historical model observations.
    """
    if timestamp_column not in base.columns:
        raise ValueError("Base frame is missing timestamp")
    if timestamp_column not in context.columns:
        raise ValueError("Context frame is missing timestamp")

    value_columns = [
        column for column in context.columns
        if column != timestamp_column
    ]
    renamed = context.rename(
        {column: f"{prefix}{column}" for column in value_columns}
    )

    return base.sort(timestamp_column).join_asof(
        renamed.sort(timestamp_column),
        on=timestamp_column,
        strategy="backward",
    )
