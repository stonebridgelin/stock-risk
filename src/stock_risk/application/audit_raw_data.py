"""用例：审计 data/raw 中的原始行情（T00 第 4 节，T00a 补充）。只报告，不修改任何数据。

各项检查都对**未更正的原始数据**运行；登记了复权价更正（I11）的代码，另对更正后的数据
重新做复权系数跳变检查，单独列出。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from stock_risk.application.audit_models import AuditReport, SymbolAudit
from stock_risk.application.market_data_loader import (
    CALENDAR_SYMBOL,
    LoadedHistory,
    MarketDataLoader,
)
from stock_risk.application.ports import (
    AuditReportWriter,
    SegmentProvider,
    UniverseProvider,
)
from stock_risk.domain.audit import (
    ADJ_FACTOR_JUMP_THRESHOLD,
    adj_factor_jumps,
    calendar_gaps,
    explicit_failures,
    manifest_mismatches,
    zero_volume_days,
)
from stock_risk.domain.segments import SymbolSegment, junction_checks


def audit_symbol(
    loaded: LoadedHistory,
    calendar: tuple[date, ...],
    segments: tuple[SymbolSegment, ...] | None,
) -> SymbolAudit:
    history, listing = loaded.raw, loaded.listing
    bars = history.bars
    action_days = {a.day for a in history.actions}
    gap_start = listing.listed_from if listing and listing.listed_from else None
    gap_end = listing.listed_until if listing and listing.listed_until else None
    gaps = calendar_gaps(bars, calendar, gap_start, gap_end)
    outside = sum(1 for b in bars if listing and not listing.contains(b.day))
    corrected_jumps = (
        tuple(adj_factor_jumps(loaded.corrected.bars, action_days)) if loaded.corrections else ()
    )
    return SymbolAudit(
        symbol=loaded.symbol,
        rows=len(bars),
        first_day=history.first_day,
        last_day=history.last_day,
        sealed_rows=loaded.sealed_rows,
        listing=listing,
        gap_start=gap_start or history.first_day,
        gap_end=gap_end or history.last_day,
        rows_outside_listing=outside,
        failures=tuple(explicit_failures(bars)),
        factor_jumps=tuple(adj_factor_jumps(bars, action_days)),
        gaps=tuple(gaps),
        zero_volume=tuple(zero_volume_days(bars)),
        junctions=tuple(junction_checks(bars, segments)) if segments else (),
        corrections=loaded.corrections,
        corrected_factor_jumps=corrected_jumps,
    )


class AuditRawDataService:
    def __init__(
        self,
        loader: MarketDataLoader,
        universe: UniverseProvider,
        segments: SegmentProvider,
        writer: AuditReportWriter,
    ) -> None:
        self._loader = loader
        self._universe = universe
        self._segments = segments
        self._writer = writer

    def run(self, run_date: date, out_root: Path) -> tuple[AuditReport, list[Path]]:
        """run_date 由调用方注入（domain 与 application 不读时钟）；输出到 out_root/<run_date>/。"""
        reader = self._loader.reader
        if not reader.exists(CALENDAR_SYMBOL):
            raise FileNotFoundError(f"缺少日历代理 {CALENDAR_SYMBOL} 的原始数据")
        calendar = tuple(b.day for b in reader.load(CALENDAR_SYMBOL).history.bars)
        segment_map = self._segments.load()
        members = list(self._universe.load())
        unknown = set(self._loader.corrections) - {m.symbol for m in members}
        if unknown:
            raise ValueError(f"更正配置中的代码不在研究样本中: {sorted(unknown)}")

        entries = reader.manifest_entries()
        report = AuditReport(
            run_date=run_date,
            manifest_sha256=reader.manifest_sha256(),
            calendar_symbol=CALENDAR_SYMBOL,
            calendar_first=calendar[0] if calendar else None,
            calendar_last=calendar[-1] if calendar else None,
            adj_factor_threshold=ADJ_FACTOR_JUMP_THRESHOLD,
            manifest_entries=len(entries),
            manifest_mismatches=manifest_mismatches(entries, reader.daily_file_hashes()),
        )
        for member in members:
            if not self._loader.exists(member.symbol):
                report.missing.append(member.symbol)
                continue
            report.symbols.append(
                audit_symbol(
                    self._loader.audit_view(member.symbol), calendar, segment_map.get(member.symbol)
                )
            )
        paths = self._writer.write(report, out_root / run_date.isoformat())
        return report, paths
