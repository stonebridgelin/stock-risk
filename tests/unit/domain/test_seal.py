from datetime import date

import pytest

from stock_risk.domain.market_data import ActionType, CorporateAction, DailyBar, PriceHistory
from stock_risk.domain.seal import (
    EXPLORATION_END,
    SealViolation,
    assert_sealed,
    truncate_to_exploration,
)


def bar(d: date) -> DailyBar:
    return DailyBar(d, 1, 2, 0.5, 1.5, 100, 1, 2, 0.5, 1.5)


def test_exploration_end_is_fixed() -> None:
    assert date(2019, 12, 31) == EXPLORATION_END


def test_truncate_drops_validation_period() -> None:
    h = PriceHistory(
        (bar(date(2019, 12, 31)), bar(date(2020, 1, 2))),
        (CorporateAction(date(2020, 1, 2), ActionType.DIVIDEND, 0.1),),
    )
    t = truncate_to_exploration(h)
    assert t.last_day == date(2019, 12, 31)
    assert t.actions == ()


def test_assert_sealed_raises_on_validation_data() -> None:
    with pytest.raises(SealViolation):
        assert_sealed(PriceHistory((bar(date(2020, 1, 2)),), ()))


def test_assert_sealed_passes_on_exploration_data() -> None:
    assert_sealed(PriceHistory((bar(date(2019, 12, 31)),), ()))
