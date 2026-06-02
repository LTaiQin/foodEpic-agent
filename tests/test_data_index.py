from food_agent.data_index import parse_seconds


def test_parse_seconds_numeric() -> None:
    assert parse_seconds(12) == 12.0
    assert parse_seconds("12.5") == 12.5


def test_parse_seconds_timestamp() -> None:
    assert parse_seconds("00:00:03.767") == 3.767
    assert parse_seconds("00:03:1.8") == 181.8


def test_parse_seconds_empty() -> None:
    assert parse_seconds("") is None
    assert parse_seconds(None) is None

