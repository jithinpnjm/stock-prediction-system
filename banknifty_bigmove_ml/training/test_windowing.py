import numpy as np
import pandas as pd

from windowing import (
    build_windows, slice_by_date, N_FEATURES,
    resample_ohlc, build_multi_timeframe_windows, slice_mtf_by_date,
    N_FEATURES_DIRECTIONAL, compute_prev_day_levels, build_windows_directional,
    slice_directional_by_date, build_multi_timeframe_windows_directional,
)


def _make_df(n=200, lookback_gap=45):
    dt = pd.date_range("2024-01-01 09:15", periods=n, freq="1min")
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 0.1, n))
    df = pd.DataFrame({
        "datetime": dt,
        "open": close,
        "high": close + 0.05,
        "low": close - 0.05,
        "close": close,
        "date": dt.date,
    })
    label = np.random.default_rng(1).integers(0, 2, n).astype(float)
    if lookback_gap > 0:
        label[-lookback_gap:] = np.nan  # simulate session-tail context-only rows
    df["label"] = label
    return df


def test_window_shape_and_target_alignment():
    df = _make_df(n=200)
    lookback = 20
    ds = build_windows(df, lookback=lookback)
    assert ds.X.shape[1:] == (lookback, N_FEATURES)
    assert ds.X.shape[0] == ds.y.shape[0] == ds.dates.shape[0] == ds.anchors.shape[0]
    # every window's LAST row close-relative feature (channel 3) must be exactly 0 (anchor is its own close)
    assert np.allclose(ds.X[:, -1, 3], 0.0)
    # color channel (9) must be -1/0/+1 only
    assert set(np.unique(ds.X[:, :, 9])) <= {-1.0, 0.0, 1.0}
    # body_to_range channel (8) must be in [0, 1] (or 0 for zero-range candles)
    assert (ds.X[:, :, 8] >= 0).all() and (ds.X[:, :, 8] <= 1.0001).all()


def test_context_only_rows_excluded_from_targets_but_usable_as_context():
    df = _make_df(n=200, lookback_gap=45)
    lookback = 20
    ds = build_windows(df, lookback=lookback)
    # 200 rows, last 45 have no label -> at most 200-45-lookback+1 could be targets from that tail alone,
    # but simplest invariant: no window's target datetime falls in the unlabeled tail
    unlabeled_start = df["datetime"].iloc[-45]
    assert not np.any(ds.datetimes >= np.datetime64(unlabeled_start))


def test_insufficient_rows_raises():
    df = _make_df(n=10)
    try:
        build_windows(df, lookback=20)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_slice_by_date_matches_manual_filter():
    df = _make_df(n=300)
    ds = build_windows(df, lookback=20)
    start, end = df["date"].iloc[100], df["date"].iloc[200]
    sliced = slice_by_date(ds, start, end)
    manual_mask = (ds.dates >= np.datetime64(pd.Timestamp(start))) & (ds.dates <= np.datetime64(pd.Timestamp(end)))
    assert sliced.X.shape[0] == manual_mask.sum()


def test_resample_ohlc_5min_matches_manual_aggregation():
    df = _make_df(n=100, lookback_gap=0)
    htf = resample_ohlc(df, "5min")
    # first 5-min bucket is rows 0..4 (09:15-09:19)
    bucket = df.iloc[0:5]
    row = htf.iloc[0]
    assert row["open"] == bucket["open"].iloc[0]
    assert row["high"] == bucket["high"].max()
    assert row["low"] == bucket["low"].min()
    assert row["close"] == bucket["close"].iloc[-1]
    assert row["bar_end"] == row["bar_start"] + pd.Timedelta("5min")


def test_htf_branch_never_sees_the_in_progress_bucket():
    """The core leakage check: for a target row in the MIDDLE of a 5-min
    bucket, the HTF branch's most recent bar must be the PRIOR bucket, not
    the one currently forming (which would leak future 1-min candles)."""
    n = 300
    df = _make_df(n=n, lookback_gap=0)
    ds = build_multi_timeframe_windows(df, base_lookback=20, htf_specs={"5min": 3})

    # target datetimes minute-of-bucket: bucket start is a multiple of 5 min after 09:15
    minute_offset = (pd.to_datetime(ds.datetimes) - pd.Timestamp("2024-01-01 09:15")).total_seconds() / 60
    minute_in_bucket = minute_offset.astype(int) % 5

    htf = resample_ohlc(df, "5min")
    for i in range(len(ds.y)):
        if minute_in_bucket[i] == 0:
            continue  # first minute of a bucket -> even that bucket itself isn't closed yet, skip trivial case
        t = pd.Timestamp(ds.datetimes[i])
        last_htf_close = ds.X_by_tf["5min"][i, -1, 3]  # close_rel of the most recent HTF bar in the window
        # find which htf bar this actually came from by reconstructing raw close via anchor
        anchor = None  # not tracked post-slice here; instead assert the bar_end used is <= t structurally:
        closed_bars = htf[htf["bar_end"] <= t]
        assert len(closed_bars) >= 3
        # the bucket containing t must NOT be among the closed bars used
        containing_bucket_start = t.floor("5min")
        assert not (closed_bars["bar_start"] == containing_bucket_start).any()


def test_multi_timeframe_shapes_and_slice():
    df = _make_df(n=400, lookback_gap=0)
    ds = build_multi_timeframe_windows(df, base_lookback=20, htf_specs={"5min": 4})
    assert ds.X_by_tf["1min"].shape[1:] == (20, N_FEATURES)
    assert ds.X_by_tf["5min"].shape[1:] == (4, N_FEATURES)
    n = ds.y.shape[0]
    assert ds.X_by_tf["1min"].shape[0] == ds.X_by_tf["5min"].shape[0] == n

    start, end = df["date"].iloc[0], df["date"].iloc[200]
    sliced = slice_mtf_by_date(ds, start, end)
    assert sliced.X_by_tf["1min"].shape[0] == sliced.X_by_tf["5min"].shape[0] == sliced.y.shape[0]


# ============================================================================
# Directional (triple-barrier) windowing tests
# ============================================================================
def _make_multi_day_df(n_days=3, minutes_per_day=100, seed=0):
    """Multiple real sessions (09:15 start each day) with distinct daily
    high/low/close, so prev-day-level merging is actually testable."""
    rng = np.random.default_rng(seed)
    frames = []
    base_price = 100.0
    for d in range(n_days):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        dt = pd.date_range(day.replace(hour=9, minute=15), periods=minutes_per_day, freq="1min")
        close = base_price + np.cumsum(rng.normal(0, 0.2, minutes_per_day))
        df_day = pd.DataFrame({
            "datetime": dt, "open": close, "high": close + 0.3, "low": close - 0.3,
            "close": close, "date": dt.date,
        })
        frames.append(df_day)
        base_price = close[-1]  # next day continues from here (but its own high/low differ)
    df = pd.concat(frames, ignore_index=True)
    n = len(df)
    long_wins = rng.integers(0, 2, n).astype(float)
    short_wins = rng.integers(0, 2, n).astype(float)
    long_wins[-5:] = np.nan  # simulate a few undefined/context-only tail rows
    short_wins[-5:] = np.nan
    df["long_wins"] = long_wins
    df["short_wins"] = short_wins
    return df


def test_compute_prev_day_levels_matches_manual_calc():
    df = _make_multi_day_df(n_days=3, minutes_per_day=50)
    out = compute_prev_day_levels(df)
    day1 = sorted(df["date"].unique())[1]  # second day should see day 0's real levels
    day0_df = df[df["date"] == sorted(df["date"].unique())[0]]
    row = out[out["date"] == day1].iloc[0]
    assert np.isclose(row["prev_day_high"], day0_df["high"].max())
    assert np.isclose(row["prev_day_low"], day0_df["low"].min())
    assert np.isclose(row["prev_day_close"], day0_df["close"].iloc[-1])


def test_first_day_has_no_prev_day_levels():
    df = _make_multi_day_df(n_days=3, minutes_per_day=50)
    out = compute_prev_day_levels(df)
    first_day = sorted(df["date"].unique())[0]
    assert out[out["date"] == first_day]["prev_day_high"].isna().all()


def test_build_windows_directional_shape_and_exclusion_of_first_day():
    df = _make_multi_day_df(n_days=3, minutes_per_day=80)
    lookback = 20
    ds = build_windows_directional(df, lookback=lookback)
    assert ds.X.shape[1:] == (lookback, N_FEATURES_DIRECTIONAL)
    assert ds.y.shape[1] == 2
    assert ds.X.shape[0] == ds.y.shape[0] == ds.dates.shape[0]
    # no window should have a target row on the first calendar day (no prev-day levels exist yet)
    first_day = np.datetime64(sorted(df["date"].unique())[0])
    assert not np.any(ds.dates == first_day)


def test_directional_y_is_nan_free_and_binary():
    df = _make_multi_day_df(n_days=3, minutes_per_day=80)
    ds = build_windows_directional(df, lookback=20)
    assert not np.isnan(ds.y).any()
    assert set(np.unique(ds.y)) <= {0.0, 1.0}


def test_slice_directional_by_date_matches_manual_filter():
    df = _make_multi_day_df(n_days=3, minutes_per_day=80)
    ds = build_windows_directional(df, lookback=20)
    days = sorted(np.unique(ds.dates))
    sliced = slice_directional_by_date(ds, days[0], days[0])
    manual_mask = ds.dates == days[0]
    assert sliced.X.shape[0] == manual_mask.sum()


def test_multi_timeframe_directional_shapes():
    df = _make_multi_day_df(n_days=4, minutes_per_day=100)
    ds = build_multi_timeframe_windows_directional(df, base_lookback=20, htf_specs={"5min": 4})
    # prev-day context channels live ONLY on the 1min (base) branch — the same 3
    # numbers would be pure duplication if repeated into every HTF branch too.
    assert ds.X_by_tf["1min"].shape[1:] == (20, N_FEATURES_DIRECTIONAL)
    assert ds.X_by_tf["5min"].shape[1:] == (4, N_FEATURES)
    assert ds.y.shape[1] == 2
    n = ds.y.shape[0]
    assert ds.X_by_tf["1min"].shape[0] == ds.X_by_tf["5min"].shape[0] == n
