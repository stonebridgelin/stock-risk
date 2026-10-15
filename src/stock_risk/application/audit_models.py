"""原始数据审计的结果对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from stock_risk.domain.audit import AdjFactorJump, ExplicitFailure, ManifestMismatch, ManifestStatus
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
    # 下载状态已确认无数据（no_data）而没有文件的代码（B-04）：只报告，不算缺失
    no_data: list[str] = field(default_factory=list)
    manifest_entries: int = 0
    manifest_mismatches: list[ManifestMismatch] = field(default_factory=list)

    @property
    def total_failures(self) -> int:
        return sum(len(s.failures) for s in self.symbols)

    @property
    def total_sealed_rows(self) -> int:
        return sum(s.sealed_rows for s in self.symbols)

    @property
    def blocking_problems(self) -> list[str]:
        """使 audit-raw 返回非零退出码的问题（T08a D-02）：明确失败、封存越界、
        MANIFEST 哈希不符或文件缺失、研究样本中有而 data/raw 中没有的代码。空表示通过。

        Gemini 第二阶段 B-02：MANIFEST.sha256 缺失或为空，或被审计证券的 CSV 不在清单中，
        同样阻断；不在清单中、也不属于被审计证券的文件只报告，不阻断。"""
        out: list[str] = []
        if self.total_failures:
            out.append(f"明确失败 {self.total_failures} 项")
        if self.total_sealed_rows:
            out.append(f"封存越界 {self.total_sealed_rows} 行")
        if self.manifest_sha256 is None:
            out.append("MANIFEST.sha256 缺失")
        elif self.manifest_entries == 0:
            out.append("MANIFEST.sha256 为空")
        audited = {f"{s.symbol}.csv" for s in self.symbols}
        for kind, label, only_audited in (
            (ManifestStatus.HASH_MISMATCH, "MANIFEST 哈希不符", False),
            (ManifestStatus.FILE_MISSING, "MANIFEST 中有而文件缺失", False),
            (ManifestStatus.NOT_IN_MANIFEST, "被审计证券不在 MANIFEST 中", True),
        ):
            files = [
                m.file
                for m in self.manifest_mismatches
                if m.status is kind and (not only_audited or m.file in audited)
            ]
            if files:
                out.append(f"{label} {len(files)} 个：{', '.join(files)}")
        if self.missing:
            out.append(f"缺少原始文件 {len(self.missing)} 个：{', '.join(self.missing)}")
        return out
