"""应用层结果对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


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
