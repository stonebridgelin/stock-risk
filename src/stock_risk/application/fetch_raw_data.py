"""用例：按研究样本下载探索期原始日线并落盘。

没有分段配置的代码：依次尝试 source_candidates，取第一个有数据的代码。
有分段配置的代码（T00 / D1）：按各段的数据源代码与区间分别下载，拼接后保存为一个文件；
任何一段失败或无数据，该代码整体判为失败，不保存部分结果。
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import UTC, date, datetime

from stock_risk.application.models import FetchOutcome, FetchReport, SegmentRecord
from stock_risk.application.ports import (
    PriceSource,
    PriceSourceError,
    RawDataRepository,
    SegmentProvider,
    UniverseProvider,
)
from stock_risk.application.progress import ProgressFn, quiet
from stock_risk.domain.market_data import PriceHistory, quality_warnings
from stock_risk.domain.seal import EXPLORATION_END, assert_sealed, truncate_to_exploration
from stock_risk.domain.segments import SegmentError, SymbolSegment, clip_history, stitch
from stock_risk.domain.universe import UniverseMember


class FetchRawDataService:
    def __init__(
        self,
        source: PriceSource,
        universe: UniverseProvider,
        repository: RawDataRepository,
        progress: ProgressFn = quiet,
        segments: SegmentProvider | None = None,
    ) -> None:
        self._source = source
        self._universe = universe
        self._repo = repository
        self._progress = progress
        self._segments = segments

    def run(
        self,
        start: date,
        end: date = EXPLORATION_END,
        *,
        force: bool = False,
        symbols: Collection[str] | None = None,
    ) -> FetchReport:
        """symbols 给定时只处理这些代码（必须属于研究样本）。"""
        if end > EXPLORATION_END:
            raise ValueError(f"结束日期不得晚于探索期截止日 {EXPLORATION_END}")
        if start > end:
            raise ValueError("开始日期晚于结束日期")

        members = list(self._universe.load())
        segment_map = self._load_segments(members)
        if symbols is not None:
            wanted = {s.upper() for s in symbols}
            unknown = wanted - {m.symbol for m in members}
            if unknown:
                raise ValueError(f"代码不在研究样本中: {sorted(unknown)}")
            members = [m for m in members if m.symbol in wanted]

        report = FetchReport(self._source.name, start, end, datetime.now(UTC))
        for member in members:
            if not force and self._repo.is_fetched(member.symbol):
                self._progress(f"[跳过] {member.symbol} 已下载")
                report.outcomes.append(
                    FetchOutcome(member.symbol, member.category, ok=True, skipped=True)
                )
                continue
            segs = segment_map.get(member.symbol)
            if segs:
                outcome = self._fetch_segmented(member, segs, start, end)
            else:
                outcome = self._fetch_one(member, start, end)
            report.outcomes.append(outcome)

        self._repo.write_summary(report)
        return report

    def _load_segments(
        self, members: list[UniverseMember]
    ) -> Mapping[str, tuple[SymbolSegment, ...]]:
        if self._segments is None:
            return {}
        segment_map = self._segments.load()
        unknown = set(segment_map) - {m.symbol for m in members}
        if unknown:
            raise ValueError(f"分段配置中的代码不在研究样本中: {sorted(unknown)}")
        return segment_map

    def _fetch_one(self, member: UniverseMember, start: date, end: date) -> FetchOutcome:
        for ticker in member.source_candidates:
            self._progress(f"[请求] {member.symbol} ← {self._source.name}:{ticker}")
            try:
                raw = self._source.fetch(ticker, start, end)
            except PriceSourceError as e:
                return self._fail(member, str(e))
            history = truncate_to_exploration(raw) if raw else None
            if history is None or not history.bars:
                self._progress("    该代码在区间内无数据")
                continue
            return self._save(member, ticker, history)
        return self._fail(member, "所有候选代码均无数据")

    def _fetch_segmented(
        self, member: UniverseMember, segs: tuple[SymbolSegment, ...], start: date, end: date
    ) -> FetchOutcome:
        parts: list[tuple[SymbolSegment, PriceHistory]] = []
        for seg in segs:
            lo = max(start, seg.start) if seg.start else start
            hi = min(end, seg.end) if seg.end else end
            if lo > hi:
                continue
            self._progress(
                f"[请求] {member.symbol} ← {self._source.name}:{seg.source_ticker}（{lo} ~ {hi}）"
            )
            try:
                raw = self._source.fetch(seg.source_ticker, lo, hi)
            except PriceSourceError as e:
                return self._fail(member, f"分段 {seg.source_ticker}: {e}")
            history = clip_history(truncate_to_exploration(raw), seg) if raw else None
            if history is None or not history.bars:
                return self._fail(member, f"分段 {seg.source_ticker} 在 {lo} ~ {hi} 无数据")
            parts.append((seg, history))
        if not parts:
            return self._fail(member, "所有分段都不在请求区间内")
        try:
            stitched = stitch(parts)
        except SegmentError as e:
            return self._fail(member, f"拼接失败: {e}")
        records = tuple(
            SegmentRecord(seg.source_ticker, h.first_day, h.last_day, len(h.bars))
            for seg, h in parts
        )
        ticker = "+".join(seg.source_ticker for seg, _ in parts)
        return self._save(member, ticker, stitched, records)

    def _fail(self, member: UniverseMember, error: str) -> FetchOutcome:
        outcome = FetchOutcome(member.symbol, member.category, ok=False, error=error)
        self._repo.record_failure(outcome)
        self._progress(f"    失败: {error}")
        return outcome

    def _save(
        self,
        member: UniverseMember,
        ticker: str,
        history: PriceHistory,
        segments: tuple[SegmentRecord, ...] = (),
    ) -> FetchOutcome:
        assert_sealed(history)
        outcome = FetchOutcome(
            symbol=member.symbol,
            category=member.category,
            ok=True,
            source_ticker=ticker,
            rows=len(history.bars),
            first_day=history.first_day,
            last_day=history.last_day,
            actions=len(history.actions),
            warnings=tuple(quality_warnings(history.bars)),
            segments=segments,
        )
        self._repo.save(member, history, outcome)
        self._progress(
            f"    OK {outcome.rows} 行 {outcome.first_day} ~ {outcome.last_day}，"
            f"公司行动 {outcome.actions} 条，质量提示 {len(outcome.warnings)} 条"
        )
        return outcome
