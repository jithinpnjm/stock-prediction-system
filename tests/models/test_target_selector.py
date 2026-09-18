from src.models.target_selector import choose_target


def test_target_selector_prefers_largest_valid_target():
    result=choose_target({200:0.80,300:0.70,400:0.60},stop_points=70,min_probability=0.55)
    assert result==400


def test_target_selector_abstains_when_probability_is_weak():
    assert choose_target({200:0.40,300:0.30}) is None
