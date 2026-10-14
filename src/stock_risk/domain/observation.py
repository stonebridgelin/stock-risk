"""三个区间与逐日"有日线 / 缺价"序列（I2、I3、I10）。

- 数据覆盖区间：文件首日至末日。
- 观察区间：起点 = max(登记的 listed_from, 数据首日)；终点 = min(探索截止日, 登记的 listed_until)。
  不取数据文件的末日：终点之前文件提前结束的，其后各交易日为缺价。
- 回放区间：起点 = max(回放下限 2009-01-01, 观察起点)；终点 = 观察终点。
各起点落在非交易日时，取其后第一个交易日。
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.listing import ListingPeriod
from stock_risk.domain.numeric import InvalidInputError
from stock_risk.domain.seal import EXPLORATION_END

REPLAY_FLOOR = date(2009, 1, 1)  # I2：回放起点下限；此前数据只用于预热


class LeftEnd(StrEnum):
    LISTING_START = "listing_start"  # 登记了上市起点，且它是观察起点的来源
    REQUEST_START = "request_start"  # 数据首日等于下载请求后的首个交易日
    FIRST_AVAILABLE = "first_available"  # 其余情况：左端覆盖未核实


class RightEnd(StrEnum):
    LISTING_END = "LISTING_END"  # 由登记的 listed_until 决定
    EXPLORATION_END = "EXPLORATION_END"  # 由探索截止日决定


@dataclass(frozen=True, slots=True)
class SymbolRanges:
    symbol: str
    coverage_start: date
    coverage_end: date
    observation_start: date
    observation_end: date
    replay_start: date
    replay_end: date
    left_end: LeftEnd
    right_end: RightEnd

    @property
    def left_coverage_verified(self) -> bool:
        return self.left_end is not LeftEnd.FIRST_AVAILABLE


def compute_ranges(
    symbol: str,
    coverage_start: date,
    coverage_end: date,
    listing: ListingPeriod | None,
    calendar: TradingCalendar,
    request_first_day: date,
    replay_floor: date = REPLAY_FLOOR,
    exploration_end: date = EXPLORATION_END,
) -> SymbolRanges:
    """request_first_day：下载请求起点之后的首个交易日（探索期为 2008-01-02）。"""
    listed_from = listing.listed_from if listing else None
    listed_until = listing.listed_until if listing else None

    if listed_from is not None and listed_from >= coverage_start:
        raw_start, left = listed_from, LeftEnd.LISTING_START
    elif coverage_start == request_first_day:
        raw_start, left = coverage_start, LeftEnd.REQUEST_START
    else:
        raw_start, left = coverage_start, LeftEnd.FIRST_AVAILABLE

    if listed_until is not None and listed_until < exploration_end:
        raw_end, right = listed_until, RightEnd.LISTING_END
    else:
        raw_end, right = exploration_end, RightEnd.EXPLORATION_END

    obs_start = calendar.first_on_or_after(raw_start)
    replay_start = calendar.first_on_or_after(max(replay_floor, raw_start))
    obs_end = _last_on_or_before(calendar, raw_end)
    if obs_start is None or obs_end is None or obs_start > obs_end:
        raise InvalidInputError(f"{symbol}: 观察区间为空（{raw_start} ~ {raw_end}）")
    if replay_start is None or replay_start > obs_end:
        raise InvalidInputError(f"{symbol}: 回放区间为空")
    return SymbolRanges(
        symbol=symbol,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        observation_start=obs_start,
        observation_end=obs_end,
        replay_start=replay_start,
        replay_end=obs_end,
        left_end=left,
        right_end=right,
    )


def _last_on_or_before(calendar: TradingCalendar, day: date) -> date | None:
    if calendar.contains(day):
        return day
    return calendar.previous(day)


@dataclass(frozen=True, slots=True)
class DayPresence:
    day: date
    has_bar: bool


def daily_presence(
    calendar: TradingCalendar, bar_days: Collection[date], ranges: SymbolRanges
) -> tuple[DayPresence, ...]:
    """观察区间内逐个交易日给出"有日线 / 缺价"。

    观察起点不早于数据首日，所以首条日线之前的日子不会出现；
    观察终点之前文件提前结束的，其后各交易日为缺价，不截掉。
    观察区间内出现日历以外日期的日线 → InvalidInputError（日历代理与数据不一致）。
    """
    have = set(bar_days)
    stray = sorted(
        d
        for d in have
        if ranges.observation_start <= d <= ranges.observation_end and not calendar.contains(d)
    )
    if stray:
        raise InvalidInputError(f"{ranges.symbol}: 日线日期不在交易日历中: {stray[:5]}")
    return tuple(
        DayPresence(d, d in have)
        for d in calendar.between(ranges.observation_start, ranges.observation_end)
    )


def missing_days(series: tuple[DayPresence, ...]) -> tuple[date, ...]:
    return tuple(p.day for p in series if not p.has_bar)
