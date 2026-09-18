from src.inference.decision import decide


def test_low_confidence_returns_no_trade():
    result = decide(
        0.34,
        0.33,
        0.33,
        min_probability=0.55,
        min_edge=0.08,
    )
    assert result.signal == 0
    assert result.reason == "abstain_probability"


def test_directional_signal_requires_edge_over_no_event():
    result = decide(
        0.15,
        0.10,
        0.75,
        target_points=200.0,
        stop_points=70.0,
        min_probability=0.55,
        min_edge=0.10,
    )
    assert result.signal == 1
    assert result.expected_value_points > 0
