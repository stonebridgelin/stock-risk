"""用例：审计 data/raw 中的原始行情（T00 第 4 节）。只报告，不修改任何数据。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from stock_risk.application.audit_models import AuditReport, SymbolAudit
from stock_risk.application.ports import (
    AuditReportWriter,
    ListingPeriodProvider,
    RawDataReader,
    SegmentProvider,
    UniverseProvider,
)
from stock_risk.domain.audit import (
    ADJ_FACTOR_JUMP_THRESHOLD,
    adj_factor_jumps,
    calendar_gaps,
    explicit_failures,
    zero_volume_days,
)
from stock_risk.domain.listing import ListingPeriod
from stock_risk.domain.market_data import PriceHistory
from stock_risk.domain.segments import SymbolSegment, junction_checks

# I1：以 SPY 的交易日作为 NYSE 交易日历的代理
CALENDAR_SYMBOL = "SPY"


def audit_symbol(
    symbol: str,
    history: PriceHistory,
    sealed_rows: int,
    calendar: tuple[date, ...],
    listing: ListingPeriod | None,
    segments: tuple[SymbolSegment, ...] | None,
) -> SymbolAudit:
    bars = history.bars
    gap_start = listing.listed_from if listing and listing.listed_from else None
    gap_end = listing.listed_until if listing and listing.listed_until else None
    gaps = calendar_gaps(bars, calendar, gap_start, gap_end)
    outside = sum(1 for b in bars if listing and not listing.contains(b.day))
    return SymbolAudit(
        symbol=symbol,
        rows=len(bars),
        first_day=history.first_day,
        last_day=history.last_day,
        sealed_rows=sealed_rows,
        listing=listing,
        gap_start=gap_start or history.first_day,
        gap_end=gap_end or history.last_day,
        rows_outside_listing=outside,
        failures=tuple(explicit_failures(bars)),
        factor_jumps=tuple(adj_factor_jumps(bars, {a.day for a in history.actions})),
        gaps=tuple(gaps),
        zero_volume=tuple(zero_volume_days(bars)),
        junctions=tuple(junction_checks(bars, segments)) if segments else (),
    )


class AuditRawDataService:
    def __init__(
        self,
        reader: RawDataReader,
        universe: UniverseProvider,
        segments: SegmentProvider,
        listings: ListingPeriodProvider,
        writer: AuditReportWriter,
    ) -> None:
        self._reader = reader
        self._universe = universe
        self._segments = segments
        self._listings = listings
        self._writer = writer

    def run(self, run_date: date, out_root: Path) -> tuple[AuditReport, list[Path]]:
        """run_date 由调用方注入（domain 与 application 不读时钟）；输出到 out_root/<run_date>/。"""
        if not self._reader.exists(CALENDAR_SYMBOL):
            raise FileNotFoundError(f"缺少日历代理 {CALENDAR_SYMBOL} 的原始数据")
        cal_file = self._reader.load(CALENDAR_SYMBOL)
        calendar = tuple(b.day for b in cal_file.history.bars)
        segment_map = self._segments.load()
        listing_map = self._listings.load()

        report = AuditReport(
            run_date=run_date,
            manifest_sha256=self._reader.manifest_sha256(),
            calendar_symbol=CALENDAR_SYMBOL,
            calendar_first=calendar[0] if calendar else None,
            calendar_last=calendar[-1] if calendar else None,
            adj_factor_threshold=ADJ_FACTOR_JUMP_THRESHOLD,
        )
        for member in self._universe.load():
            if not self._reader.exists(member.symbol):
                report.missing.append(member.symbol)
                continue
            raw = cal_file if member.symbol == CALENDAR_SYMBOL else self._reader.load(member.symbol)
            report.symbols.append(
                audit_symbol(
                    member.symbol,
                    raw.history,
                    raw.sealed_rows,
                    calendar,
                    listing_map.get(member.symbol),
                    segment_map.get(member.symbol),
                )
            )
        paths = self._writer.write(report, out_root / run_date.isoformat())
        return report, paths
