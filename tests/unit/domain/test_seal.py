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


def test_reference_objects_reject_sealed_dates() -> None:
    """T08a D-02：日历、上市期间、分段、更正记录构造时检查日期不晚于 2019-12-31。"""
    from stock_risk.domain.calendar import TradingCalendar
    from stock_risk.domain.corrections import CorrectionMethod, PriceCorrection
    from stock_risk.domain.listing import ListingPeriod
    from stock_risk.domain.segments import SymbolSegment

    ok, bad = date(2019, 12, 31), date(2020, 1, 2)
    TradingCalendar.from_days([ok])
    ListingPeriod("X", listed_until=ok, source_url="u", verified_on=date(2026, 10, 13))
    SymbolSegment("X", "X", start=ok)
    method = next(iter(CorrectionMethod))
    PriceCorrection("X", ok, ok, method, "PM-T", "e")
    with pytest.raises(SealViolation):
        TradingCalendar.from_days([ok, bad])
    with pytest.raises(SealViolation):
        ListingPeriod("X", listed_until=bad, source_url="u")
    with pytest.raises(SealViolation):
        ListingPeriod("X", listed_from=bad, source_url="u")
    with pytest.raises(SealViolation):
        SymbolSegment("X", "X", end=bad)
    with pytest.raises(SealViolation):
        PriceCorrection("X", ok, bad, method, "PM-T", "e")
