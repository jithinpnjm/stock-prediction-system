"""Prior-day Bank Nifty options/futures OI context, from NSE's daily F&O
bhavcopy. This is a genuinely different information source from every
other feature in this codebase (all of which derive from Bank Nifty's
own OHLCV price action) -- it reflects options-market positioning
(put/call OI skew, futures OI buildup, max pain) that price action alone
cannot see. Only the PRIOR day's EOD snapshot is used: the bhavcopy for
day T is only published after T's market close, so it is causally valid
context for every bar of session T+1 (and all following days until the
next snapshot lands), never for day T itself.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

_RAW_COLS = [
    "fut_oi",
    "fut_oi_chg",
    "opt_pcr_oi",
    "opt_pcr_volume",
    "opt_total_ce_oi_chg",
    "opt_total_pe_oi_chg",
    "opt_max_pain_strike",
    "opt_atm_ce_oi",
    "opt_atm_pe_oi",
]


def add_options_context_features(
    df: pl.DataFrame,
    bhavcopy_path: str | Path = "data/bronze/banknifty_fo_daily.parquet",
) -> pl.DataFrame:
    path = Path(bhavcopy_path)
    if not path.exists():
        return df

    daily = pl.read_parquet(path).sort("date")
    if daily.height == 0:
        return df

    eps = 1e-9
    daily = daily.with_columns(
        (pl.col("opt_atm_pe_oi") / (pl.col("opt_atm_ce_oi") + eps)).alias("_atm_pcr"),
        (pl.col("fut_oi_chg") / (pl.col("fut_oi") - pl.col("fut_oi_chg") + eps)).alias(
            "_fut_oi_chg_pct"
        ),
        ((pl.col("opt_max_pain_strike") - pl.col("fut_close")) / (pl.col("fut_close") + eps)).alias(
            "_max_pain_dist_pct"
        ),
    )
    # shift by one row = prior trading day's snapshot, causally valid for "today"
    daily = daily.with_columns(
        pl.col("date").alias("_session_date"),
        pl.col("opt_pcr_oi").shift(1).alias("f_prev_opt_pcr_oi"),
        pl.col("opt_pcr_volume").shift(1).alias("f_prev_opt_pcr_volume"),
        pl.col("_atm_pcr").shift(1).alias("f_prev_opt_atm_pcr"),
        pl.col("_fut_oi_chg_pct").shift(1).alias("f_prev_fut_oi_chg_pct"),
        pl.col("_max_pain_dist_pct").shift(1).alias("f_prev_max_pain_dist_pct"),
        (pl.col("opt_pcr_oi") - pl.col("opt_pcr_oi").shift(1))
        .shift(1)
        .alias("f_prev_opt_pcr_oi_change"),
    ).select(
        [
            "_session_date",
            "f_prev_opt_pcr_oi",
            "f_prev_opt_pcr_volume",
            "f_prev_opt_atm_pcr",
            "f_prev_fut_oi_chg_pct",
            "f_prev_max_pain_dist_pct",
            "f_prev_opt_pcr_oi_change",
        ]
    )

    out = df.with_columns(pl.col("timestamp").dt.date().alias("_session_date"))
    out = out.join(daily, on="_session_date", how="left").drop("_session_date")
    return out
