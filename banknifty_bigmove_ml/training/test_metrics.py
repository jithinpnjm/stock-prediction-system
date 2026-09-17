import numpy as np

from metrics import compute_directional_strategy_metrics


def test_takes_the_higher_confidence_direction_above_threshold():
    p_long = np.array([0.8, 0.2, 0.9])
    p_short = np.array([0.1, 0.7, 0.3])
    long_wins = np.array([1.0, 0.0, 1.0])
    short_wins = np.array([0.0, 1.0, 0.0])
    m = compute_directional_strategy_metrics(p_long, p_short, long_wins, short_wins, threshold=0.5)
    assert m.n_trades == 3
    assert m.n_long == 2  # rows 0 and 2
    assert m.n_short == 1  # row 1
    assert m.win_rate == 1.0  # all three chosen-direction outcomes were wins


def test_skips_when_neither_side_clears_threshold():
    p_long = np.array([0.3, 0.4])
    p_short = np.array([0.2, 0.35])
    long_wins = np.array([1.0, 1.0])
    short_wins = np.array([0.0, 0.0])
    m = compute_directional_strategy_metrics(p_long, p_short, long_wins, short_wins, threshold=0.5)
    assert m.n_trades == 0
    assert np.isnan(m.win_rate)


def test_skips_ambiguous_ties_rather_than_guessing():
    p_long = np.array([0.7])
    p_short = np.array([0.7])  # exact tie, both above threshold
    long_wins = np.array([1.0])
    short_wins = np.array([1.0])
    m = compute_directional_strategy_metrics(p_long, p_short, long_wins, short_wins, threshold=0.5)
    assert m.n_trades == 0  # never guesses on a tie


def test_expected_points_matches_the_100_50_breakeven_math():
    # 4 trades, all long, exactly the 33.3% breakeven win rate (1 of 3 wins... use 4 trades for exactness)
    p_long = np.array([0.9, 0.9, 0.9])
    p_short = np.array([0.1, 0.1, 0.1])
    long_wins = np.array([1.0, 0.0, 0.0])  # win_rate = 1/3
    short_wins = np.array([0.0, 0.0, 0.0])
    m = compute_directional_strategy_metrics(p_long, p_short, long_wins, short_wins, threshold=0.5,
                                              target_pts=100.0, stop_pts=50.0)
    win_rate = 1 / 3
    expected = win_rate * 100.0 - (1 - win_rate) * 50.0
    assert abs(m.expected_points_per_trade - expected) < 1e-9
    assert abs(m.expected_points_per_trade - 0.0) < 1e-9  # 1/3 is exactly the breakeven point at 100/50


def test_long_and_short_win_rates_tracked_separately():
    p_long = np.array([0.9, 0.9, 0.1, 0.1])
    p_short = np.array([0.1, 0.1, 0.9, 0.9])
    long_wins = np.array([1.0, 0.0, 0.0, 0.0])
    short_wins = np.array([0.0, 0.0, 1.0, 1.0])
    m = compute_directional_strategy_metrics(p_long, p_short, long_wins, short_wins, threshold=0.5)
    assert m.long_win_rate == 0.5   # 1 of 2 long trades won
    assert m.short_win_rate == 1.0  # 2 of 2 short trades won
