from datetime import date

import pytest

from stock_risk.domain.numeric import InvalidInputError
from stock_risk.domain.raw_text import RawActionText, RawBarText, parse_bar, parse_history

D = date(2019, 12, 2)


def text_bar(day: date = D, **over: str) -> RawBarText:
    base = dict(
        open="10",
        high="11",
        low="9",
        close="10.5",
        volume="1000.0",
        adj_open="5.0000005",
        adj_high="5.5",
        adj_low="4.5",
        adj_close="5.2500015",
    )
    base.update(over)
    return RawBarText(day, **base)


def test_parse_bar_quantizes_every_price() -> None:
    b = parse_bar("X", text_bar())
    assert (b.adj_open, b.adj_close) == (5.0, 5.250002)
    assert b.volume == 1000.0


@pytest.mark.parametrize(
    "over",
    [
        {"high": "8"},  # 原始价 高 < 低
        {"adj_close": "6"},  # 复权收盘高于复权最高
        {"close": "nan"},
        {"low": "0"},
        {"volume": "inf"},
    ],
)
def test_parse_bar_explicit_failures(over: dict[str, str]) -> None:
    with pytest.raises(InvalidInputError):
        parse_bar("X", text_bar(**over))


def test_parse_history_requires_ascending_and_known_action_types() -> None:
    with pytest.raises(InvalidInputError):
        parse_history("X", (text_bar(date(2019, 12, 3)), text_bar(D)), ())
    with pytest.raises(InvalidInputError):
        parse_history("X", (text_bar(),), (RawActionText(D, "rights", "1"),))
    h = parse_history("X", (text_bar(),), (RawActionText(D, "dividend", "0.1"),))
    assert h.actions[0].value == 0.1
