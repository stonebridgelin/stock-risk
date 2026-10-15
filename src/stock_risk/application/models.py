"""应用层结果对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    """分段拼接中一段实际取得的数据：数据源代码、首末日期、行数。"""

    source_ticker: str
    first_day: date | None
    last_day: date | None
    rows: int


@dataclass(frozen=True, slots=True)
class FetchOutcome:
    symbol: str
    category: str
    ok: bool
    source_ticker: str | None = None
    rows: int = 0
    first_day: date | None = None
    last_day: date | None = None
    actions: int = 0
    warnings: tuple[str, ...] = ()
    error: str | None = None
    skipped: bool = False
    segments: tuple[SegmentRecord, ...] = ()
    no_data: bool = False  # 数据源没有该代码（404 / not found），不是调用失败（T08d-1）

    @property
    def state(self) -> str:
        return "ok" if self.ok else ("no_data" if self.no_data else "failed")


@dataclass(slots=True)
class FetchReport:
    source_name: str
    start: date
    end: date
    started_at: datetime
    outcomes: list[FetchOutcome] = field(default_factory=list)

    @property
    def succeeded(self) -> list[FetchOutcome]:
        return [o for o in self.outcomes if o.ok]

    @property
    def failed(self) -> list[FetchOutcome]:
        return [o for o in self.outcomes if not o.ok]
