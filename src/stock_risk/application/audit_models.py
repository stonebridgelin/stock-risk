"""原始数据审计的结果对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from stock_risk.domain.audit import AdjFactorJump, ExplicitFailure, ManifestMismatch
from stock_risk.domain.corrections import PriceCorrection
from stock_risk.domain.listing import ListingPeriod
from stock_risk.domain.segments import JunctionCheck


@dataclass(frozen=True, slots=True)
class SymbolAudit:
    symbol: str
    rows: int
    first_day: date | None
    last_day: date | None
    sealed_rows: int
    listing: ListingPeriod | None
    gap_start: date | None  # 缺口检查区间（上市期间配置生效时以其为界）
    gap_end: date | None
    rows_outside_listing: int
    failures: tuple[ExplicitFailure, ...]
    factor_jumps: tuple[AdjFactorJump, ...]
    gaps: tuple[date, ...]
    zero_volume: tuple[date, ...]
    junctions: tuple[JunctionCheck, ...]
    corrections: tuple[PriceCorrection, ...] = ()  # 已登记的复权价更正（I11）
    corrected_factor_jumps: tuple[AdjFactorJump, ...] = ()  # 更正后数据的系数跳变


@dataclass(slots=True)
class AuditReport:
    run_date: date
    manifest_sha256: str | None
    calendar_symbol: str
    calendar_first: date | None
    calendar_last: date | None
    adj_factor_threshold: float
    symbols: list[SymbolAudit] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # 研究样本中有、data/raw 中没有的代码
    manifest_entries: int = 0
    manifest_mismatches: list[ManifestMismatch] = field(default_factory=list)

    @property
    def total_failures(self) -> int:
        return sum(len(s.failures) for s in self.symbols)

    @property
    def total_sealed_rows(self) -> int:
        return sum(s.sealed_rows for s in self.symbols)
