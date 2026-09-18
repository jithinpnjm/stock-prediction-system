from src.data.schemas import DataContract


def test_default_contract():
    c=DataContract()
    assert c.timezone=="Asia/Kolkata"
    assert c.source_timestamp_semantics=="start"
    assert c.evaluation_open=="09:30"
