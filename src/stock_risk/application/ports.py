"""应用层端口（Protocol）。基础设施层实现它们，应用层只依赖这些抽象。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Protocol

from stock_risk.application.models import FetchOutcome, FetchReport
from stock_risk.domain.market_data import PriceHistory
from stock_risk.domain.universe import UniverseMember


class PriceSourceError(RuntimeError):
    """数据源调用失败（网络、鉴权、限频重试耗尽等），区别于“代码不存在”。"""


class PriceSource(Protocol):
    name: str

    def fetch(self, ticker: str, start: date, end: date) -> PriceHistory | None:
        """返回区间内的日线；代码在数据源不存在时返回 None；调用失败抛 PriceSourceError。"""
        ...


class UniverseProvider(Protocol):
    def load(self) -> Sequence[UniverseMember]: ...


class RawDataRepository(Protocol):
    def is_fetched(self, symbol: str) -> bool:
        """该代码是否已成功落盘（用于断点续传）。"""
        ...

    def save(
        self, member: UniverseMember, history: PriceHistory, outcome: FetchOutcome
    ) -> None: ...

    def record_failure(self, outcome: FetchOutcome) -> None: ...

    def write_summary(self, report: FetchReport) -> None:
        """写出合并的公司行动、README 与校验清单。"""
        ...
