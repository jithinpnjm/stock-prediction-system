"""
Automated spot-check for label_logic.label_session — replaces one-off manual
eyeballing with a repeatable test, since a labeling bug here silently taints
every downstream training run.

Run: python3 -m pytest banknifty_bigmove_ml/data_pipeline/test_label_logic.py -q
"""
import numpy as np

from label_logic import label_session


def test_no_move_stays_zero():
    n = 50
    close = np.full(n, 100.0)
    high = np.full(n, 100.05)
    low = np.full(n, 99.95)
    label = label_session(high, low, close, horizon=45, pct=0.003)
    assert label[0] == 0.0
    assert np.isnan(label[6])  # only 43 rows left after t=6 (< horizon=45)


def test_upward_touch_flags_one():
    n = 100
    close = np.full(n, 100.0)
    high = np.full(n, 100.05)
    low = np.full(n, 99.95)
    # candle 10 spikes to +0.3% -> should flag every t where 10 is in [t+1, t+45]
    high[10] = 100.0 * 1.003
    label = label_session(high, low, close, horizon=45, pct=0.003)
    assert label[0] == 1.0     # window t=0 -> [1..45] includes 10
    assert label[9] == 1.0     # window t=9 -> [10..54] includes 10
    assert label[10] == 0.0    # window t=10 -> [11..55], excludes candle 10 itself


def test_downward_touch_flags_one():
    n = 100
    close = np.full(n, 100.0)
    high = np.full(n, 100.05)
    low = np.full(n, 99.95)
    low[20] = 100.0 * 0.997
    label = label_session(high, low, close, horizon=45, pct=0.003)
    assert label[0] == 1.0
    assert label[19] == 1.0
    assert label[20] == 0.0


def test_short_session_all_nan():
    n = 30  # shorter than horizon
    close = np.full(n, 100.0)
    high = np.full(n, 100.05)
    low = np.full(n, 99.95)
    label = label_session(high, low, close, horizon=45, pct=0.003)
    assert np.all(np.isnan(label))


def test_last_horizon_rows_are_nan():
    n = 100
    close = np.full(n, 100.0)
    high = np.full(n, 100.05)
    low = np.full(n, 99.95)
    label = label_session(high, low, close, horizon=45, pct=0.003)
    assert not np.isnan(label[n - 45 - 1])  # last labeled row
    assert np.isnan(label[n - 45])          # first of the trailing unlabeled rows
    assert np.all(np.isnan(label[n - 45:]))
