"""应用层端口（Protocol）。基础设施层实现它们，应用层只依赖这些抽象。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from stock_risk.application.audit_models import AuditReport
from stock_risk.application.models import FetchOutcome, FetchReport
from stock_risk.domain.corrections import PriceCorrection
from stock_risk.domain.listing import ListingPeriod
from stock_risk.domain.market_data import PriceHistory
from stock_risk.domain.segments import SymbolSegment
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


class SegmentProvider(Protocol):
    def load(self) -> Mapping[str, tuple[SymbolSegment, ...]]:
        """按 symbol 返回分段配置；没有分段的代码不出现在结果中。"""
        ...


class ListingPeriodProvider(Protocol):
    def load(self) -> Mapping[str, ListingPeriod]: ...


class PriceCorrectionProvider(Protocol):
    def load(self) -> Mapping[str, tuple[PriceCorrection, ...]]:
        """按 symbol 返回已登记的复权价更正（I11）；没有更正的代码不出现在结果中。"""
        ...


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


@dataclass(frozen=True, slots=True)
class RawFile:
    """读取到的原始数据。sealed_rows：晚于探索期截止日的行数，这些行不解析、不返回。"""

    symbol: str
    history: PriceHistory
    sealed_rows: int


class RawDataReader(Protocol):
    def exists(self, symbol: str) -> bool: ...

    def load(self, symbol: str) -> RawFile: ...

    def manifest_sha256(self) -> str | None:
        """MANIFEST.sha256 文件本身的 SHA-256，作为数据版本标识；没有清单时为 None。"""
        ...

    def manifest_entries(self) -> Mapping[str, str]:
        """清单中的条目：日线文件名 → SHA-256。"""
        ...

    def daily_file_hashes(self) -> Mapping[str, str]:
        """目录中实际的日线文件名 → SHA-256。"""
        ...


class AuditReportWriter(Protocol):
    def write(self, report: AuditReport, out_dir: Path) -> list[Path]:
        """把审计结果写到 out_dir，返回写出的文件。"""
        ...
