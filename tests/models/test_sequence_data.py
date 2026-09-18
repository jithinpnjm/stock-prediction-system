import numpy as np

from src.models.sequence_data import build_sequences


FIVE_MIN_NS = 5 * 60 * 1_000_000_000


def test_sequence_windows_are_causal():
    x = np.arange(20 * 2, dtype=float).reshape(20, 2)
    y = np.zeros(20, dtype=int)
    ts = np.arange(20, dtype=np.int64) * FIVE_MIN_NS
    batch = build_sequences(x, y, ts, sequence_length=5)
    assert batch.X.shape == (16, 5, 2)
    assert np.array_equal(batch.timestamps, ts[4:])
    assert np.array_equal(batch.X[0, -1], x[4])


def test_sequence_windows_do_not_cross_sessions():
    x = np.arange(8 * 2, dtype=float).reshape(8, 2)
    y = np.zeros(8, dtype=int)
    ts = np.array(
        [
            0,
            FIVE_MIN_NS,
            2 * FIVE_MIN_NS,
            3 * FIVE_MIN_NS,
            100 * FIVE_MIN_NS,
            101 * FIVE_MIN_NS,
            102 * FIVE_MIN_NS,
            103 * FIVE_MIN_NS,
        ],
        dtype=np.int64,
    )
    session_dates = np.array(
        ["2026-01-05"] * 4 + ["2026-01-06"] * 4,
        dtype="U10",
    )

    batch = build_sequences(
        x,
        y,
        ts,
        sequence_length=3,
        session_dates=session_dates,
    )

    assert np.array_equal(
        batch.timestamps,
        np.array([2 * FIVE_MIN_NS, 3 * FIVE_MIN_NS, 102 * FIVE_MIN_NS, 103 * FIVE_MIN_NS]),
    )
