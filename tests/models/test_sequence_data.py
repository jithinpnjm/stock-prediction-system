import numpy as np

from src.models.sequence_data import build_sequences

BAR_NS = 300 * 1_000_000_000


def test_sequence_windows_are_causal():
    x = np.arange(20 * 2, dtype=float).reshape(20, 2)
    y = np.zeros(20, dtype=int)
    ts = np.arange(20) * BAR_NS
    batch = build_sequences(x, y, ts, sequence_length=5)
    assert batch.X.shape == (16, 5, 2)
    assert np.array_equal(batch.timestamps, ts[4:])
    assert np.array_equal(batch.X[0, -1], x[4])


def test_sequence_skips_windows_spanning_a_timestamp_gap():
    """A window whose bars are not all exactly 5 minutes apart must be
    dropped rather than silently treated as one continuous sequence
    (e.g. rows removed upstream by null-filtering, or a session
    boundary crossing)."""
    x = np.arange(10 * 2, dtype=float).reshape(10, 2)
    y = np.zeros(10, dtype=int)
    ts = np.arange(10) * BAR_NS
    # Introduce a large gap between bar index 4 and 5.
    ts[5:] += 10 * BAR_NS

    batch = build_sequences(x, y, ts, sequence_length=5)
    # Windows ending at indices 4..8 (0-indexed) would span the gap for
    # any window that includes both sides of it; only windows entirely
    # before or entirely after the gap should survive.
    produced_end_positions = {np.where(ts == t)[0][0] for t in batch.timestamps}
    for end_idx in produced_end_positions:
        window_start = end_idx - 4
        assert window_start >= 5 or end_idx <= 4, (
            f"window ending at {end_idx} spans the gap and should have been dropped"
        )
