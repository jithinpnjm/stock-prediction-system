"""Historical NSE F&O bhavcopy ingestion: per-day, per-strike Bank Nifty
futures/options OI, volume, and price -- a genuinely different data
source from the OHLCV price-action features used everywhere else in this
codebase. Freely available as a static daily archive file, no login/
cookie gate (unlike the interactive nseindia.com API, which 403s from
non-browser/datacenter clients)."""

from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass
from datetime import date

import polars as pl
import requests

_LEGACY_URL = (
    "https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
    "{year}/{mon}/fo{day:02d}{mon}{year}bhav.csv.zip"
)
# NSE migrated the F&O bhavcopy to this "UDiFF" format/naming around
# 2024-07-08; the legacy URL above 404s for every date from that point
# on (confirmed: the legacy backfill silently stopped at 2024-07-05 with
# 544 consecutive 404s, not real missing days). Different column names,
# same underlying data -- normalized to the legacy schema below so
# parse_banknifty_daily doesn't need to know which era it's reading.
_UDIFF_URL = (
    "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip"
)
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


@dataclass(frozen=True)
class FetchResult:
    trading_date: date
    raw: pl.DataFrame | None  # None means no data for this date (holiday etc.)


_UDIFF_INSTRUMENT_MAP = {"IDF": "FUTIDX", "IDO": "OPTIDX", "STF": "FUTSTK", "STO": "OPTSTK"}
_UDIFF_RENAME = {
    "TckrSymb": "SYMBOL",
    "FinInstrmTp": "INSTRUMENT",
    "StrkPric": "STRIKE_PR",
    "OptnTp": "OPTION_TYP",
    "ClsPric": "CLOSE",
    "OpnIntrst": "OPEN_INT",
    "ChngInOpnIntrst": "CHG_IN_OI",
    "TtlTradgVol": "CONTRACTS",
}


def _normalize_udiff(raw: pl.DataFrame) -> pl.DataFrame:
    out = raw.rename(_UDIFF_RENAME)
    out = out.with_columns(
        pl.col("INSTRUMENT").replace(_UDIFF_INSTRUMENT_MAP),
        pl.col("XpryDt")
        .str.strptime(pl.Date, "%Y-%m-%d")
        .dt.strftime("%d-%b-%Y")
        .alias("EXPIRY_DT"),
    )
    return out


def _legacy_url(d: date) -> str:
    mon = d.strftime("%b").upper()
    return _LEGACY_URL.format(year=d.year, mon=mon, day=d.day)


def _udiff_url(d: date) -> str:
    return _UDIFF_URL.format(ymd=d.strftime("%Y%m%d"))


def _get_zip_csv(url: str, timeout: int) -> pl.DataFrame | None:
    resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = zf.namelist()[0]
        return pl.read_csv(
            zf.read(name), infer_schema_length=None, schema_overrides={"STRIKE_PR": pl.Float64}
        )


def fetch_bhavcopy(d: date, *, timeout: int = 20, max_attempts: int = 4) -> FetchResult:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            raw = _get_zip_csv(_legacy_url(d), timeout)
            if raw is not None:
                return FetchResult(d, raw)
            raw = _get_zip_csv(_udiff_url(d), timeout)
            if raw is not None:
                return FetchResult(d, _normalize_udiff(raw))
            return FetchResult(d, None)
        except Exception as exc:  # noqa: BLE001 -- retry on any transient failure
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to fetch bhavcopy for {d}: {last_exc}") from last_exc


def parse_banknifty_daily(raw: pl.DataFrame, d: date) -> dict | None:
    """Aggregate one day's raw bhavcopy rows into BANKNIFTY-level features.

    All derived from that day's own EOD snapshot -- available only after
    market close, so these are usable as a *prior-day* context feature
    (same causal contract as f_prev_day_* elsewhere), not an intraday one.
    """
    cols = {c.strip(): c for c in raw.columns}
    raw = raw.rename({v: k for k, v in cols.items()})
    bn = raw.filter(pl.col("SYMBOL") == "BANKNIFTY")
    if bn.height == 0:
        return None

    bn = bn.with_columns(
        pl.col("EXPIRY_DT").str.strptime(pl.Date, "%d-%b-%Y").alias("_expiry_date")
    )
    futs = bn.filter(pl.col("INSTRUMENT") == "FUTIDX").sort("_expiry_date")
    opts = bn.filter(pl.col("INSTRUMENT") == "OPTIDX")
    if opts.height == 0 or futs.height == 0:
        return None

    # nearest-expiry future = current month contract
    near_fut = futs.head(1)
    fut_close = float(near_fut["CLOSE"][0])
    fut_oi = float(near_fut["OPEN_INT"][0])
    fut_oi_chg = float(near_fut["CHG_IN_OI"][0])

    # nearest expiry among options (current weekly/monthly)
    near_expiry = opts["_expiry_date"].min()
    near_opts = opts.filter(pl.col("_expiry_date") == near_expiry)

    ce = near_opts.filter(pl.col("OPTION_TYP") == "CE")
    pe = near_opts.filter(pl.col("OPTION_TYP") == "PE")
    total_ce_oi = float(ce["OPEN_INT"].sum())
    total_pe_oi = float(pe["OPEN_INT"].sum())
    total_ce_vol = float(ce["CONTRACTS"].sum())
    total_pe_vol = float(pe["CONTRACTS"].sum())
    total_ce_oi_chg = float(ce["CHG_IN_OI"].sum())
    total_pe_oi_chg = float(pe["CHG_IN_OI"].sum())

    pcr_oi = total_pe_oi / (total_ce_oi + 1e-9)
    pcr_vol = total_pe_vol / (total_ce_vol + 1e-9)

    # max pain: strike where total option-writer payout is minimized
    strikes = near_opts["STRIKE_PR"].unique().sort()
    max_pain_strike = None
    if strikes.len() > 0:
        best_strike, best_pain = None, float("inf")
        ce_by_strike = ce.group_by("STRIKE_PR").agg(pl.col("OPEN_INT").sum().alias("oi"))
        pe_by_strike = pe.group_by("STRIKE_PR").agg(pl.col("OPEN_INT").sum().alias("oi"))
        ce_map = dict(zip(ce_by_strike["STRIKE_PR"].to_list(), ce_by_strike["oi"].to_list()))
        pe_map = dict(zip(pe_by_strike["STRIKE_PR"].to_list(), pe_by_strike["oi"].to_list()))
        strike_list = strikes.to_list()
        for s in strike_list:
            pain = sum(
                max(s - k, 0) * ce_map.get(k, 0.0) + max(k - s, 0) * pe_map.get(k, 0.0)
                for k in strike_list
            )
            if pain < best_pain:
                best_pain = pain
                best_strike = s
        max_pain_strike = best_strike

    # OI concentration near ATM (within 500 pts of futures close)
    atm_ce = ce.filter((pl.col("STRIKE_PR") - fut_close).abs() <= 500)
    atm_pe = pe.filter((pl.col("STRIKE_PR") - fut_close).abs() <= 500)
    atm_ce_oi = float(atm_ce["OPEN_INT"].sum())
    atm_pe_oi = float(atm_pe["OPEN_INT"].sum())

    return {
        "date": d,
        "fut_close": fut_close,
        "fut_oi": fut_oi,
        "fut_oi_chg": fut_oi_chg,
        "opt_pcr_oi": pcr_oi,
        "opt_pcr_volume": pcr_vol,
        "opt_total_ce_oi": total_ce_oi,
        "opt_total_pe_oi": total_pe_oi,
        "opt_total_ce_oi_chg": total_ce_oi_chg,
        "opt_total_pe_oi_chg": total_pe_oi_chg,
        "opt_max_pain_strike": float(max_pain_strike) if max_pain_strike is not None else None,
        "opt_atm_ce_oi": atm_ce_oi,
        "opt_atm_pe_oi": atm_pe_oi,
        "opt_near_expiry": str(near_expiry),
    }
