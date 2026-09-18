"""
data_downloader_new.py
-----------------------
Pulls BANKNIFTY spot (NSE:NIFTYBANK-INDEX) 1-minute candles from Fyers for the
last N years and keeps a single running CSV up to date.

Resumable / retryable by design:
  - Every fetched chunk is merged into the CSV and saved immediately, so a
    crash or Ctrl-C never loses already-downloaded data.
  - Re-running the script does NOT start over. It scans the existing CSV for
    missing or incomplete trading days (day granularity) and only fetches
    those date ranges (candle granularity — a day with a suspiciously low
    candle count is treated as incomplete and re-pulled).
  - Confirmed no-data days (weekends already skipped; holidays) are cached in
    a small sidecar file so repeat runs don't keep re-requesting them.
  - Rate-limit (-429) and transient errors are retried with backoff; fatal
    error codes abort a chunk without derailing the rest of the run.

Run this any time to fill in whatever is missing:
    python3 data_downloader_new.py
"""
import json
import os
import time
import warnings
from datetime import datetime, timedelta
from datetime import time as dt_time

import pandas as pd
import pytz

import fyers_auth

warnings.filterwarnings("ignore")

IST = pytz.timezone("Asia/Kolkata")
MARKET_CLOSE_TIME = dt_time(15, 30)

# ============================================================================
# CONFIGURATION
# ============================================================================
SYMBOL = "NSE:NIFTYBANK-INDEX"
RESOLUTION = "1"  # 1-minute candles
YEARS_BACK = 5
CHUNK_DAYS = 60  # Fyers cap per history() call for sub-daily resolutions
INTER_CHUNK_SLEEP = 1.5  # seconds between chunk requests (rate-limit safety)
MAX_ATTEMPTS = 3
MIN_CANDLES_FOR_COMPLETE_DAY = 50  # below this, a trading day is treated as a gap

FATAL_CODES = {494, 400, -101, -300}

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "banknifty_spot_1m.csv")
NO_DATA_FILE = os.path.join(DATA_DIR, "banknifty_spot_1m_no_data_days.json")
os.makedirs(DATA_DIR, exist_ok=True)

END_DATE = datetime.today().date()
START_DATE = (pd.Timestamp(END_DATE) - pd.DateOffset(years=YEARS_BACK)).date()


# ============================================================================
# STATE (existing CSV + confirmed no-data days)
# ============================================================================
def load_existing():
    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 100:
        try:
            return pd.read_csv(OUTPUT_FILE, parse_dates=["datetime"])
        except Exception:
            pass
    return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])


def load_no_data_days():
    if os.path.exists(NO_DATA_FILE):
        try:
            with open(NO_DATA_FILE) as f:
                return {datetime.strptime(d, "%Y-%m-%d").date() for d in json.load(f)}
        except Exception:
            return set()
    return set()


def save_no_data_days(days):
    with open(NO_DATA_FILE, "w") as f:
        json.dump(sorted(d.strftime("%Y-%m-%d") for d in days), f, indent=2)


def save_merged(existing, new_data):
    if new_data.empty:
        return existing
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined = (
        combined.drop_duplicates("datetime", keep="last")
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    combined.to_csv(OUTPUT_FILE, index=False)
    return combined


def today_is_stale(day):
    """True if `day` is today and today's session hasn't closed yet."""
    now_ist = datetime.now(IST)
    if day != now_ist.date():
        return False
    close_cutoff = IST.localize(datetime.combine(day, MARKET_CLOSE_TIME))
    return now_ist < close_cutoff


# ============================================================================
# GAP DETECTION (day + candle granularity)
# ============================================================================
def find_gap_ranges(df, no_data_days, start_date, end_date):
    """Weekdays in [start_date, end_date] that are missing, incomplete, or
    (for today) still mid-session — grouped into contiguous fetch ranges."""
    if df.empty:
        good_days = set()
    else:
        counts = df["datetime"].dt.date.value_counts()
        good_days = set(counts[counts >= MIN_CANDLES_FOR_COMPLETE_DAY].index)

    missing_days = []
    d = start_date
    while d <= end_date:
        if d.weekday() < 5:  # Mon-Fri only; weekends never trade
            if d in no_data_days:
                pass
            elif today_is_stale(d) or d not in good_days:
                missing_days.append(d)
        d += timedelta(days=1)

    if not missing_days:
        return []

    ranges = []
    range_start = prev = missing_days[0]
    for day in missing_days[1:]:
        if (day - prev).days <= 3 and (day - range_start).days < CHUNK_DAYS:
            prev = day
            continue
        ranges.append((range_start, prev))
        range_start = prev = day
    ranges.append((range_start, prev))
    return ranges


# ============================================================================
# DOWNLOAD LOGIC
# ============================================================================
def fetch_range(fyers_client, start_date, end_date):
    """Fetch [start_date, end_date] in <=CHUNK_DAYS windows.
    Returns (df, failed_windows) — failed_windows are (start,end) sub-ranges
    that exhausted retries, so the caller never mis-marks them as no-data."""
    downloaded_dfs = []
    failed_windows = []
    curr_start = pd.Timestamp(start_date)
    final_end = pd.Timestamp(end_date)

    while curr_start <= final_end:
        curr_end = min(curr_start + timedelta(days=CHUNK_DAYS - 1), final_end)
        data = {
            "symbol": SYMBOL,
            "resolution": RESOLUTION,
            "date_format": "1",
            "range_from": curr_start.strftime("%Y-%m-%d"),
            "range_to": curr_end.strftime("%Y-%m-%d"),
            "cont_flag": "1",
        }

        chunk_success = False
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = fyers_client.history(data=data)

                if not isinstance(response, dict):
                    print(
                        f"   ⚠️ Invalid API response (attempt {attempt + 1}/{MAX_ATTEMPTS}): {response}"
                    )
                    time.sleep(5 * (attempt + 1))
                    continue

                if response.get("code") == -429:
                    wait = 15 * (attempt + 1)
                    print(
                        f"   ⏳ Rate limit. Waiting {wait}s (attempt {attempt + 1}/{MAX_ATTEMPTS})..."
                    )
                    time.sleep(wait)
                    continue

                if response.get("s") == "no_data":
                    chunk_success = True
                    break

                if response.get("s") == "ok" and "candles" in response:
                    cdf = pd.DataFrame(
                        response["candles"],
                        columns=["ts", "open", "high", "low", "close", "volume"],
                    )
                    if not cdf.empty:
                        cdf["datetime"] = (
                            pd.to_datetime(cdf["ts"], unit="s")
                            .dt.tz_localize("UTC")
                            .dt.tz_convert("Asia/Kolkata")
                            .dt.tz_localize(None)
                        )
                        downloaded_dfs.append(cdf)
                    chunk_success = True
                    break

                err_code = response.get("code", "?")
                err_msg = response.get("message", str(response))
                print(
                    f"   ⚠️ API error (attempt {attempt + 1}/{MAX_ATTEMPTS}): [{err_code}] {err_msg}"
                )
                if err_code in FATAL_CODES:
                    print(f"   ❌ Fatal error [{err_code}] — aborting this chunk.")
                    break
                if "limit" in str(err_msg).lower():
                    time.sleep(30 * (attempt + 1))
                else:
                    time.sleep(5 * (attempt + 1))
            except Exception as e:
                print(f"   ⚠️ Exception (attempt {attempt + 1}/{MAX_ATTEMPTS}): {e}")
                time.sleep(5 * (attempt + 1))

        if not chunk_success:
            print(
                f"   ❌ Gave up on {curr_start.date()} → {curr_end.date()} after {MAX_ATTEMPTS} attempts"
            )
            failed_windows.append((curr_start.date(), curr_end.date()))

        time.sleep(INTER_CHUNK_SLEEP)
        curr_start = curr_end + timedelta(days=1)

    if downloaded_dfs:
        combined = (
            pd.concat(downloaded_dfs, ignore_index=True)
            .drop("ts", axis=1)
            .drop_duplicates("datetime")
            .sort_values("datetime")
            .reset_index(drop=True)
        )
        return combined[
            ["datetime", "open", "high", "low", "close", "volume"]
        ], failed_windows

    return pd.DataFrame(
        columns=["datetime", "open", "high", "low", "close", "volume"]
    ), failed_windows


def run():
    fyers = fyers_auth.get_fyers_session()
    print("✅ Fyers Authentication Successful.")
    print(f"📅 Date range : {START_DATE} → {END_DATE}")
    print(f"📂 Output     : {OUTPUT_FILE}")

    existing = load_existing()
    no_data_days = load_no_data_days()
    print(
        f"   Existing rows : {len(existing)}"
        + (
            f" ({existing['datetime'].min()} → {existing['datetime'].max()})"
            if not existing.empty
            else ""
        )
    )
    print(f"   Known no-data days cached : {len(no_data_days)}")

    gap_ranges = find_gap_ranges(existing, no_data_days, START_DATE, END_DATE)
    if not gap_ranges:
        print("\n✅ Already up to date — nothing to fetch.")
        return

    print(f"\n⬇️  {len(gap_ranges)} range(s) to fetch:")
    for s, e in gap_ranges:
        print(f"   - {s} → {e}")

    for i, (range_start, range_end) in enumerate(gap_ranges, start=1):
        print(f"\n[{i}/{len(gap_ranges)}] Fetching {range_start} → {range_end}...")
        new_data, failed_windows = fetch_range(fyers, range_start, range_end)

        existing = save_merged(existing, new_data)
        print(
            f"   💾 Saved — {len(new_data)} new rows → {len(existing)} total rows on disk"
        )

        failed_days = set()
        for fs, fe in failed_windows:
            d = fs
            while d <= fe:
                failed_days.add(d)
                d += timedelta(days=1)

        fetched_days = (
            set(new_data["datetime"].dt.date) if not new_data.empty else set()
        )
        today = datetime.now(IST).date()
        d = range_start
        while d <= range_end:
            if (
                d.weekday() < 5
                and d not in failed_days
                and d not in fetched_days
                and d != today
            ):
                no_data_days.add(d)
            d += timedelta(days=1)
        save_no_data_days(no_data_days)

    print("\n📊 Final check...")
    remaining = find_gap_ranges(
        load_existing(), load_no_data_days(), START_DATE, END_DATE
    )
    if remaining:
        print(
            f"   ⚠️ {len(remaining)} range(s) still incomplete (rate-limited or fatal errors above) — rerun to retry:"
        )
        for s, e in remaining:
            print(f"      - {s} → {e}")
    else:
        print("   ✅ All caught up.")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"❌ Critical Failure: {e}")
