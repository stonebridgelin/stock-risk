"""用例：选参集批量下载 `fetch-batch`（T08d-1 第 2 节）。

在 fetch_raw_data 的基础上扩展，复用同一个数据源适配器、封存三道关（结束日不晚于截止日、
返回后截断、写盘前检查）、断点续传（_fetch_status.json）与 MANIFEST。

额度保护：每小时上限换算为请求间隔（由组装入口设置适配器）；每日请求数、每月新代码数达到上限
即正常退出；连续 3 次 429 保存进度后退出（退出码非 0）。额度计数持久化，按 UTC 日期、月份记账。
可随时中断：每个代码完成即落盘并保存额度，再次运行跳过已完成（成功或无数据）的代码。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Protocol

from stock_risk.application.fetch_raw_data import record_failure, save_history
from stock_risk.application.models import FetchOutcome, FetchReport
from stock_risk.application.ports import (
    CountingPriceSource,
    PriceSourceError,
    RateLimitError,
    RawDataRepository,
)
from stock_risk.application.progress import ProgressFn, quiet
from stock_risk.domain.seal import EXPLORATION_END, truncate_to_exploration
from stock_risk.domain.universe import Role, UniverseMember

CATEGORY = "selection"
DONE_STATES = ("ok", "no_data")


@dataclass(frozen=True, slots=True)
class BatchItem:
    """下载清单的一行：保存用的代码、数据源代码与区间（to 不得晚于 2019-12-31）。"""

    symbol: str
    source_ticker: str
    start: date
    end: date

    def __post_init__(self) -> None:
        if not self.symbol or not self.source_ticker:
            raise ValueError("symbol 与 source_ticker 不能为空")
        if self.end > EXPLORATION_END:
            raise ValueError(f"{self.symbol}: to 不得晚于探索期截止日 {EXPLORATION_END}")
        if self.start > self.end:
            raise ValueError(f"{self.symbol}: from 晚于 to")

    def member(self) -> UniverseMember:
        return UniverseMember(self.symbol, (self.source_ticker,), CATEGORY, Role.SIGNAL_TARGET)


@dataclass(frozen=True, slots=True)
class QuotaLimits:
    per_hour: int = 50
    per_day: int = 1000
    unique_per_month: int = 500
    max_symbols: int | None = None

    def __post_init__(self) -> None:
        for name in ("per_hour", "per_day", "unique_per_month"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} 必须为正整数")
        if self.max_symbols is not None and self.max_symbols < 1:
            raise ValueError("max_symbols 必须为正整数")


@dataclass(slots=True)
class QuotaState:
    """按 UTC 日期记每日请求数，按 UTC 月份记本月请求过的代码。"""

    day: str = ""
    day_requests: int = 0
    month: str = ""
    month_tickers: list[str] = field(default_factory=list)

    def roll(self, now: datetime) -> None:
        d, m = now.astimezone(UTC).strftime("%Y-%m-%d"), now.astimezone(UTC).strftime("%Y-%m")
        if d != self.day:
            self.day, self.day_requests = d, 0
        if m != self.month:
            self.month, self.month_tickers = m, []

    def is_new(self, ticker: str) -> bool:
        return ticker.upper() not in self.month_tickers

    def add_ticker(self, ticker: str) -> None:
        if self.is_new(ticker):
            self.month_tickers.append(ticker.upper())

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "day_requests": self.day_requests,
            "month": self.month,
            "month_tickers": list(self.month_tickers),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> QuotaState:
        d = d or {}
        return cls(
            str(d.get("day", "")),
            int(d.get("day_requests", 0)),
            str(d.get("month", "")),
            [str(t) for t in d.get("month_tickers", [])],
        )


class QuotaStore(Protocol):
    def load(self) -> dict[str, Any] | None: ...

    def save(self, state: dict[str, Any]) -> None: ...


@dataclass(slots=True)
class BatchReport:
    total: int
    done_now: list[FetchOutcome] = field(default_factory=list)
    failed_now: list[FetchOutcome] = field(default_factory=list)
    skipped_done: int = 0
    stop_reason: str | None = None  # max_symbols / per_day / unique_per_month / rate_limited
    cumulative_done: int = 0
    day_requests: int = 0
    month_unique: int = 0
    estimated_runs: int = 0

    @property
    def remaining(self) -> int:
        return self.total - self.cumulative_done

    @property
    def rate_limited(self) -> bool:
        return self.stop_reason == "rate_limited"


class FetchBatchService:
    def __init__(
        self,
        source: CountingPriceSource,
        repository: RawDataRepository,
        quota: QuotaStore,
        now: Callable[[], datetime],
        progress: ProgressFn = quiet,
    ) -> None:
        self._source = source
        self._repo = repository
        self._quota = quota
        self._now = now
        self._progress = progress

    def run(self, items: Sequence[BatchItem], limits: QuotaLimits) -> BatchReport:
        if not items:
            raise ValueError("下载清单为空")
        symbols = [i.symbol for i in items]
        if len(set(symbols)) != len(symbols):
            raise ValueError("下载清单中有重复的 symbol")
        state = QuotaState.from_dict(self._quota.load())
        report = BatchReport(len(items))
        fetch_report = FetchReport(
            self._source.name, min(i.start for i in items), EXPLORATION_END, self._now()
        )
        try:
            for item in items:
                if self._repo.state_of(item.symbol) in DONE_STATES:
                    report.skipped_done += 1
                    continue
                report.stop_reason = self._stop_reason(item, state, limits, report)
                if report.stop_reason is not None:
                    break
                outcome = self._fetch(item, state)
                if outcome is None:
                    report.stop_reason = "rate_limited"
                    break
                fetch_report.outcomes.append(outcome)
                (report.done_now if outcome.state in DONE_STATES else report.failed_now).append(
                    outcome
                )
        finally:
            self._quota.save(state.to_dict())
            self._repo.write_summary(fetch_report)
        return self._finish(report, items, state, limits)

    def _stop_reason(
        self, item: BatchItem, state: QuotaState, limits: QuotaLimits, report: BatchReport
    ) -> str | None:
        processed = len(report.done_now) + len(report.failed_now)
        if limits.max_symbols is not None and processed >= limits.max_symbols:
            return "max_symbols"
        state.roll(self._now())
        if state.day_requests >= limits.per_day:
            return "per_day"
        if state.is_new(item.source_ticker) and len(state.month_tickers) >= limits.unique_per_month:
            return "unique_per_month"
        return None

    def _fetch(self, item: BatchItem, state: QuotaState) -> FetchOutcome | None:
        """一个代码一次请求取全区间。返回 None 表示限频用尽（未记录该代码的状态）。"""
        member = item.member()
        self._progress(f"[请求] {item.symbol} ← {self._source.name}:{item.source_ticker}")
        state.add_ticker(item.source_ticker)
        before = self._source.request_count
        try:
            raw = self._source.fetch(item.source_ticker, item.start, item.end)
        except RateLimitError:
            self._progress("    连续 3 次 429：额度已用尽，稍后再运行")
            return None
        except PriceSourceError as e:
            return record_failure(self._repo, member, str(e), self._progress)
        finally:
            state.day_requests += self._source.request_count - before
            self._quota.save(state.to_dict())
        history = truncate_to_exploration(raw) if raw else None
        if history is None or not history.bars:
            return record_failure(
                self._repo, member, "数据源无该代码或区间内无数据", self._progress, no_data=True
            )
        return save_history(self._repo, member, item.source_ticker, history, self._progress)

    def _finish(
        self,
        report: BatchReport,
        items: Sequence[BatchItem],
        state: QuotaState,
        limits: QuotaLimits,
    ) -> BatchReport:
        report.cumulative_done = sum(
            1 for i in items if self._repo.state_of(i.symbol) in DONE_STATES
        )
        report.day_requests = state.day_requests
        report.month_unique = len(state.month_tickers)
        per_run = min(limits.per_day, limits.unique_per_month, limits.max_symbols or limits.per_day)
        report.estimated_runs = math.ceil(report.remaining / per_run) if report.remaining else 0
        return report
