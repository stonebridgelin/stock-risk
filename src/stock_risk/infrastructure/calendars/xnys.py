"""独立交易所日历：exchange_calendars 的 XNYS（I13，实现 check_calendar.ReferenceCalendar）。

exchange_calendars 是开发依赖（pyproject dev 组，固定版本），只用于 check-calendar 核对，
因此在使用时才导入。
"""

from __future__ import annotations

from datetime import date
from importlib.metadata import version


class ExchangeCalendarsXnys:
    library = "exchange_calendars"
    calendar_name = "XNYS"

    def __init__(self) -> None:
        self.version = version("exchange_calendars")

    def sessions(self, start: date, end: date) -> tuple[date, ...]:
        import exchange_calendars as xcals

        cal = xcals.get_calendar(self.calendar_name, start=start.isoformat(), end=end.isoformat())
        return tuple(ts.date() for ts in cal.sessions_in_range(start.isoformat(), end.isoformat()))
