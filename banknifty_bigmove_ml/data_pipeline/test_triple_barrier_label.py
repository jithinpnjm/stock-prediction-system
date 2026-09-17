import numpy as np

from triple_barrier_label import triple_barrier_session


def _flat_session(n, base=50000.0, minute_start=555):  # 555 = 9:15
    close = np.full(n, base)
    high = close + 1.0
    low = close - 1.0
    minute_of_day = minute_start + np.arange(n)
    return high, low, close, minute_of_day


def test_long_wins_when_target_hit_before_stop():
    n = 20
    high, low, close, mod = _flat_session(n)
    high[5] = close[5] + 100  # long target hit at candle 5
    long_wins, short_wins = triple_barrier_session(high, low, close, mod)
    assert long_wins[0] == 1.0
    assert short_wins[0] == 0.0  # short's target (-100) never hit -> loss


def test_long_loses_when_stop_hit_before_target():
    n = 20
    high, low, close, mod = _flat_session(n)
    low[3] = close[3] - 50    # long stop hit at candle 3
    high[10] = close[10] + 100  # target would hit later, but stop already happened
    long_wins, short_wins = triple_barrier_session(high, low, close, mod)
    assert long_wins[0] == 0.0


def test_tie_break_favors_stop_when_both_hit_same_candle():
    n = 20
    high, low, close, mod = _flat_session(n)
    high[4] = close[4] + 100   # target
    low[4] = close[4] - 50     # AND stop, same candle -> conservative: stop wins
    long_wins, short_wins = triple_barrier_session(high, low, close, mod)
    assert long_wins[0] == 0.0


def test_timeout_with_neither_barrier_hit_is_a_loss_not_nan():
    n = 20
    high, low, close, mod = _flat_session(n)  # never moves
    long_wins, short_wins = triple_barrier_session(high, low, close, mod)
    assert long_wins[0] == 0.0
    assert short_wins[0] == 0.0


def test_entries_at_or_after_cutoff_are_nan_not_scored():
    n = 20
    high, low, close, mod = _flat_session(n, minute_start=14 * 60 + 55)  # 14:55 start
    high[10] = close[10] + 100  # would be a long win if scored
    long_wins, short_wins = triple_barrier_session(high, low, close, mod, entry_cutoff_minute=15 * 60)
    # candles at/after 15:00 (index where minute_of_day >= 900) must be NaN
    cutoff_idx = np.searchsorted(mod, 15 * 60)
    assert np.all(np.isnan(long_wins[cutoff_idx:]))
    assert np.all(np.isnan(short_wins[cutoff_idx:]))
    # candles before cutoff are still scored normally
    assert not np.isnan(long_wins[0])


def test_last_candle_of_session_always_nan_no_forward_data():
    n = 20
    high, low, close, mod = _flat_session(n)
    long_wins, short_wins = triple_barrier_session(high, low, close, mod)
    assert np.isnan(long_wins[-1])
    assert np.isnan(short_wins[-1])


def test_short_wins_symmetric_to_long():
    n = 20
    high, low, close, mod = _flat_session(n)
    low[6] = close[6] - 100  # short target hit
    long_wins, short_wins = triple_barrier_session(high, low, close, mod)
    assert short_wins[0] == 1.0
    assert long_wins[0] == 0.0  # long's target (+100) never hit


def test_no_leakage_entry_cannot_resolve_using_its_own_candle():
    n = 20
    high, low, close, mod = _flat_session(n)
    # entry candle 0 itself has a huge high, but that's the ENTRY candle, not a forward one
    high[0] = close[0] + 1000
    long_wins, short_wins = triple_barrier_session(high, low, close, mod)
    assert long_wins[0] == 0.0  # must not count candle 0's own high as a forward touch
