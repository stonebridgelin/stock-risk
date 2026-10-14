"""用例：用独立的交易所日历核对 SPY 日历代理（I13）。验证工具，结果不进入 domain。

有任何差异都只报告，由产品经理裁定；不得自行改用其中一方。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from stock_risk.application.market_data_loader import CALENDAR_SYMBOL, MarketDataLoader

CHECK_START = date(2008, 1, 2)
CHECK_END = date(2019, 12, 31)


class ReferenceCalendar(Protocol):
    """独立交易所日历（例如 exchange_calendars 的 XNYS）。"""

    library: str
    version: str
    calendar_name: str

    def sessions(self, start: date, end: date) -> tuple[date, ...]: ...


@dataclass(frozen=True, slots=True)
class CalendarCheckResult:
    start: date
    end: date
    proxy_symbol: str
    proxy_file_sha256: str | None
    proxy_days: int
    reference_library: str
    reference_version: str
    reference_calendar: str
    reference_days: int
    only_in_proxy: tuple[date, ...]
    only_in_reference: tuple[date, ...]

    @property
    def identical(self) -> bool:
        return not self.only_in_proxy and not self.only_in_reference


class CalendarCheckWriter(Protocol):
    def write(self, result: CalendarCheckResult, out_dir: Path) -> list[Path]: ...


def compare_calendars(
    proxy: tuple[date, ...], reference: tuple[date, ...], start: date, end: date
) -> tuple[tuple[date, ...], tuple[date, ...]]:
    """返回 (只在代理中, 只在参照中)，只比较 [start, end]。"""
    p = {d for d in proxy if start <= d <= end}
    r = {d for d in reference if start <= d <= end}
    return tuple(sorted(p - r)), tuple(sorted(r - p))


class CheckCalendarService:
    def __init__(
        self, loader: MarketDataLoader, reference: ReferenceCalendar, writer: CalendarCheckWriter
    ) -> None:
        self._loader = loader
        self._reference = reference
        self._writer = writer

    def run(self, run_date: date, out_root: Path) -> tuple[CalendarCheckResult, list[Path]]:
        proxy = self._loader.calendar().days
        reference = self._reference.sessions(CHECK_START, CHECK_END)
        only_proxy, only_ref = compare_calendars(proxy, reference, CHECK_START, CHECK_END)
        result = CalendarCheckResult(
            start=CHECK_START,
            end=CHECK_END,
            proxy_symbol=CALENDAR_SYMBOL,
            proxy_file_sha256=self._loader.reader.daily_file_hashes().get(f"{CALENDAR_SYMBOL}.csv"),
            proxy_days=sum(1 for d in proxy if CHECK_START <= d <= CHECK_END),
            reference_library=self._reference.library,
            reference_version=self._reference.version,
            reference_calendar=self._reference.calendar_name,
            reference_days=len(reference),
            only_in_proxy=only_proxy,
            only_in_reference=only_ref,
        )
        return result, self._writer.write(result, out_root / run_date.isoformat())
