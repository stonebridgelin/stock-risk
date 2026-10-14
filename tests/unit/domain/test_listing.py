from datetime import date

import pytest

from stock_risk.domain.listing import (
    ListingPeriod,
    clip_bars_to_listing,
    clip_history_to_listing,
)
from stock_risk.domain.market_data import ActionType, CorporateAction, DailyBar, PriceHistory


def bar(d: date) -> DailyBar:
    return DailyBar(d, 1, 2, 0.5, 1.5, 100, 1, 2, 0.5, 1.5)


DAYS = [date(2018, 10, 22), date(2018, 10, 23), date(2018, 10, 24)]
URL = "https://www.sec.gov/x"


def test_clip_to_listed_until_inclusive() -> None:
    p = ListingPeriod("SHLD", listed_until=date(2018, 10, 23), source_url=URL)
    assert [b.day for b in clip_bars_to_listing([bar(d) for d in DAYS], p)] == DAYS[:2]


def test_clip_to_listed_from_inclusive() -> None:
    p = ListingPeriod("SMH", listed_from=date(2018, 10, 23), source_url=URL)
    assert [b.day for b in clip_bars_to_listing([bar(d) for d in DAYS], p)] == DAYS[1:]


def test_no_period_or_unbounded_period_keeps_everything() -> None:
    bars = [bar(d) for d in DAYS]
    assert clip_bars_to_listing(bars, None) == tuple(bars)
    assert clip_bars_to_listing(bars, ListingPeriod("X", note="待核实")) == tuple(bars)


def test_clip_history_also_clips_actions() -> None:
    h = PriceHistory(
        tuple(bar(d) for d in DAYS), (CorporateAction(DAYS[2], ActionType.DIVIDEND, 0.1),)
    )
    p = ListingPeriod("SHLD", listed_until=DAYS[1], source_url=URL)
    clipped = clip_history_to_listing(h, p)
    assert clipped.last_day == DAYS[1] and clipped.actions == ()


def test_dates_require_source_url() -> None:
    with pytest.raises(ValueError):
        ListingPeriod("SHLD", listed_until=DAYS[1])


def test_reversed_period_rejected() -> None:
    with pytest.raises(ValueError):
        ListingPeriod("X", DAYS[2], DAYS[0], source_url=URL)
