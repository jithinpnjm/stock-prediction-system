import pytest

from src.inference.decision import decide


def test_no_trade_when_no_event_dominates():
    result=decide(0.2,0.6,0.2)
    assert result.signal==0


def test_directional_decision_requires_edge_over_null():
    result=decide(0.55,0.40,0.05,min_probability=0.50,min_edge=0.10)
    assert result.signal==-1


def test_invalid_probabilities_rejected():
    with pytest.raises(ValueError):
        decide(-0.1,0.4,0.7)
