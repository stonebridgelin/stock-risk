from datetime import date, timedelta

import pytest

from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.listing import ListingPeriod
from stock_risk.domain.numeric import InvalidInputError
from stock_risk.domain.observation import (
    LeftEnd,
    RightEnd,
    compute_ranges,
    daily_presence,
    missing_days,
)


def weekdays(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


CAL = TradingCalendar.from_days(weekdays(date(2008, 1, 2), date(2019, 12, 31)))
REQ = date(2008, 1, 2)
URL = "https://x"


def test_calendar_queries() -> None:
    assert CAL.previous(date(2019, 12, 30)) == date(2019, 12, 27)  # 周一 → 上周五
    assert CAL.previous(date(2008, 1, 2)) is None
    assert CAL.first_on_or_after(date(2009, 1, 1)) == date(2009, 1, 1)
    assert CAL.between(date(2019, 12, 27), date(2019, 12, 31)) == (
        date(2019, 12, 27),
        date(2019, 12, 30),
        date(2019, 12, 31),
    )
    assert not CAL.contains(date(2019, 12, 28))


def test_calendar_rejects_unsorted() -> None:
    with pytest.raises(InvalidInputError):
        TradingCalendar((date(2019, 1, 3), date(2019, 1, 2)))


def test_request_start_and_exploration_end() -> None:
    r = compute_ranges("AAPL", REQ, date(2019, 12, 31), None, CAL, REQ)
    assert (r.left_end, r.right_end) == (LeftEnd.REQUEST_START, RightEnd.EXPLORATION_END)
    assert r.observation_start == REQ and r.observation_end == date(2019, 12, 31)
    assert r.replay_start == date(2009, 1, 1)  # 本测试日历中 2009-01-01 是工作日


def test_replay_start_snaps_to_first_trading_day() -> None:
    cal = TradingCalendar.from_days(d for d in CAL.days if d != date(2009, 1, 1))
    r = compute_ranges("AAPL", REQ, date(2019, 12, 31), None, cal, REQ)
    assert r.replay_start == date(2009, 1, 2)


def test_first_available_when_late_start_without_listing() -> None:
    r = compute_ranges("TSLA", date(2010, 6, 29), date(2019, 12, 31), None, CAL, REQ)
    assert r.left_end is LeftEnd.FIRST_AVAILABLE and not r.left_coverage_verified
    assert r.replay_start == date(2010, 6, 29)


def test_listing_start_and_listing_end() -> None:
    smh = ListingPeriod("SMH", listed_from=date(2011, 12, 21), source_url=URL)
    r = compute_ranges("SMH", REQ, date(2019, 12, 31), smh, CAL, REQ)
    assert r.left_end is LeftEnd.LISTING_START and r.observation_start == date(2011, 12, 21)
    shld = ListingPeriod("SHLD", listed_until=date(2018, 10, 23), source_url=URL)
    r = compute_ranges("SHLD", REQ, date(2019, 12, 31), shld, CAL, REQ)
    assert r.right_end is RightEnd.LISTING_END and r.observation_end == date(2018, 10, 23)


def test_listing_from_before_data_start_is_not_the_source() -> None:
    p = ListingPeriod("X", listed_from=date(2005, 1, 3), source_url=URL)
    r = compute_ranges("X", date(2010, 6, 29), date(2019, 12, 31), p, CAL, REQ)
    assert r.left_end is LeftEnd.FIRST_AVAILABLE and r.observation_start == date(2010, 6, 29)


def test_tail_missing_until_observation_end() -> None:
    # 上市股票文件止于 2019-12-20：12-23 至 12-31 各交易日为缺价
    days = weekdays(date(2019, 12, 2), date(2019, 12, 20))
    r = compute_ranges("X", days[0], days[-1], None, CAL, REQ)
    series = daily_presence(CAL, days, r)
    assert missing_days(series) == tuple(weekdays(date(2019, 12, 23), date(2019, 12, 31)))
    assert series[0].has_bar and series[0].day == days[0]


def test_inner_gap_and_no_missing_before_first_bar() -> None:
    days = [d for d in weekdays(date(2017, 11, 6), date(2017, 11, 10)) if d != date(2017, 11, 8)]
    r = compute_ranges("SHLD", days[0], days[-1], None, CAL, REQ)
    series = daily_presence(CAL, days, r)
    assert missing_days(series)[0] == date(2017, 11, 8)
    assert all(p.day >= days[0] for p in series)


def test_bar_outside_calendar_rejected() -> None:
    days = [date(2019, 12, 27), date(2019, 12, 28)]  # 12-28 是周六
    r = compute_ranges("X", days[0], days[-1], None, CAL, REQ)
    with pytest.raises(InvalidInputError):
        daily_presence(CAL, days, r)


def test_empty_observation_rejected() -> None:
    p = ListingPeriod("X", listed_until=date(2009, 1, 5), source_url=URL)
    with pytest.raises(InvalidInputError):
        compute_ranges("X", date(2010, 1, 4), date(2019, 12, 31), p, CAL, REQ)
