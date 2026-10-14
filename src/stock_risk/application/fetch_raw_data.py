"""用例：按研究样本下载探索期原始日线并落盘。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime

from stock_risk.application.models import FetchOutcome, FetchReport
from stock_risk.application.ports import (
    PriceSource,
    PriceSourceError,
    RawDataRepository,
    UniverseProvider,
)
from stock_risk.domain.market_data import PriceHistory, quality_warnings
from stock_risk.domain.seal import EXPLORATION_END, assert_sealed, truncate_to_exploration
from stock_risk.domain.universe import UniverseMember

ProgressFn = Callable[[str], None]


class FetchRawDataService:
    def __init__(
        self,
        source: PriceSource,
        universe: UniverseProvider,
        repository: RawDataRepository,
        progress: ProgressFn = print,
    ) -> None:
        self._source = source
        self._universe = universe
        self._repo = repository
        self._progress = progress

    def run(self, start: date, end: date = EXPLORATION_END, *, force: bool = False) -> FetchReport:
        if end > EXPLORATION_END:
            raise ValueError(f"结束日期不得晚于探索期截止日 {EXPLORATION_END}")
        if start > end:
            raise ValueError("开始日期晚于结束日期")

        report = FetchReport(self._source.name, start, end, datetime.now(UTC))
        for member in self._universe.load():
            if not force and self._repo.is_fetched(member.symbol):
                self._progress(f"[跳过] {member.symbol} 已下载")
                report.outcomes.append(
                    FetchOutcome(member.symbol, member.category, ok=True, skipped=True)
                )
                continue
            outcome = self._fetch_one(member, start, end)
            report.outcomes.append(outcome)

        self._repo.write_summary(report)
        return report

    def _fetch_one(self, member: UniverseMember, start: date, end: date) -> FetchOutcome:
        for ticker in member.source_candidates:
            self._progress(f"[请求] {member.symbol} ← {self._source.name}:{ticker}")
            try:
                raw = self._source.fetch(ticker, start, end)
            except PriceSourceError as e:
                outcome = FetchOutcome(member.symbol, member.category, ok=False, error=str(e))
                self._repo.record_failure(outcome)
                self._progress(f"    失败: {e}")
                return outcome
            history = truncate_to_exploration(raw) if raw else None
            if history is None or not history.bars:
                self._progress("    该代码在区间内无数据")
                continue
            return self._save(member, ticker, history)

        outcome = FetchOutcome(
            member.symbol, member.category, ok=False, error="所有候选代码均无数据"
        )
        self._repo.record_failure(outcome)
        return outcome

    def _save(self, member: UniverseMember, ticker: str, history: PriceHistory) -> FetchOutcome:
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
        )
        self._repo.save(member, history, outcome)
        self._progress(
            f"    OK {outcome.rows} 行 {outcome.first_day} ~ {outcome.last_day}，"
            f"公司行动 {outcome.actions} 条，质量提示 {len(outcome.warnings)} 条"
        )
        return outcome
