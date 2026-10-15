"""交易日历（I1）：以 SPY 的交易日作为 NYSE 交易日历的代理。

domain 只使用这个代理日历；与独立交易所日历的核对（I13）是验证工具，不在 domain。
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from stock_risk.domain.numeric import InvalidInputError
from stock_risk.domain.seal import assert_dates_not_sealed


@dataclass(frozen=True, slots=True)
class TradingCalendar:
    """升序、无重复的交易日序列。"""

    days: tuple[date, ...]

    def __post_init__(self) -> None:
        if not self.days:
            raise InvalidInputError("交易日历为空")
        if any(a >= b for a, b in zip(self.days, self.days[1:], strict=False)):
            raise InvalidInputError("交易日历未严格升序（重复或乱序）")
        assert_dates_not_sealed("交易日历", self.days[-1])  # 升序，只需检查最后一日

    @classmethod
    def from_days(cls, days: Iterable[date]) -> TradingCalendar:
        return cls(tuple(days))

    @property
    def first(self) -> date:
        return self.days[0]

    @property
    def last(self) -> date:
        return self.days[-1]

    def contains(self, day: date) -> bool:
        i = bisect.bisect_left(self.days, day)
        return i < len(self.days) and self.days[i] == day

    def previous(self, day: date) -> date | None:
        """day 之前最近的一个交易日（不含 day）；日历中没有更早的交易日时为 None。"""
        i = bisect.bisect_left(self.days, day)
        return self.days[i - 1] if i > 0 else None

    def first_on_or_after(self, day: date) -> date | None:
        i = bisect.bisect_left(self.days, day)
        return self.days[i] if i < len(self.days) else None

    def between(self, start: date, end: date) -> tuple[date, ...]:
        """[start, end] 内的交易日（含两端）。"""
        lo = bisect.bisect_left(self.days, start)
        hi = bisect.bisect_right(self.days, end)
        return self.days[lo:hi]
