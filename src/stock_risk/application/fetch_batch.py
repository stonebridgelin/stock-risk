"""用例：选参集批量下载 `fetch-batch`（T08d-1 第 2 节）。

在 fetch_raw_data 的基础上扩展，复用同一个数据源适配器、封存三道关（结束日不晚于截止日、
返回后截断、写盘前检查）、断点续传（_fetch_status.json）与 MANIFEST。

额度保护：每小时上限换算为请求间隔（由组装入口设置适配器）；每日请求数、每月新代码数达到上限
即正常退出；连续 3 次 429 保存进度后退出（退出码非 0）。额度计数持久化，按 UTC 日期、月份记账。
可随时中断：每个代码完成即落盘并保存额度，再次运行跳过已完成（成功或无数据）的代码。

请求记录（T10-M0b 第 6 节，F-06 ②）：目标目录中的 `_request_ledger.json` 记录数据源名称、每个代码的
source_ticker / start / end，以及完成结果（ok 带价格文件与公司行动文件的 SHA-256；no_data 只记身份；
pending 为已追加但未完成；failed 为上次请求报错）。运行开始时、发出任何请求之前判定目录能否复用：
旧目录（有状态或数据文件、没有请求记录）、数据源不同、身份不同、文件缺失或哈希不符、来历不明的文件
一律拒绝（UnverifiedDirectory），0 次请求，目录中的文件不改动。"已完成"只由 is_done 判定，
同时用于跳过、cumulative_done 与 remaining；旧 `_fetch_status.json` 不再单独决定"已完成"。
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
PENDING, FAILED = "pending", "failed"  # 请求记录中的未完成、失败


class UnverifiedDirectory(ValueError):
    """目标目录不能复用（T10-M0b 第 6 节）。信息写明代码、哪一项不一致或"旧目录没有请求记录"。"""


class BatchRepository(RawDataRepository, Protocol):
    """fetch-batch 的存储端口：在原始数据仓库之上增加请求记录与目录核对。"""

    def load_ledger(self) -> dict[str, Any] | None: ...

    def save_ledger(self, ledger: dict[str, Any]) -> None: ...

    def has_legacy_files(self) -> bool:
        """目录中是否有状态文件或数据文件（.csv）。"""
        ...

    def has_status(self, symbol: str) -> bool:
        """`_fetch_status.json` 中是否有该代码的记录。"""
        ...

    def file_hashes(self, symbol: str) -> dict[str, str | None]:
        """价格文件与公司行动文件的 SHA-256；不存在为 None。键 price、actions。"""
        ...


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
        repository: BatchRepository,
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
        ledger = self._verify_directory(items)  # 被拒绝时抛出，0 次请求、不写任何文件
        self._repo.save_ledger(ledger)
        state = QuotaState.from_dict(self._quota.load())
        report = BatchReport(len(items))
        fetch_report = FetchReport(
            self._source.name, min(i.start for i in items), EXPLORATION_END, self._now()
        )
        try:
            for item in items:
                if self.is_done(ledger, item):
                    report.skipped_done += 1
                    continue
                report.stop_reason = self._stop_reason(item, state, limits, report)
                if report.stop_reason is not None:
                    break
                if item.symbol not in ledger["items"]:  # 请求之前追加进记录
                    ledger["items"][item.symbol] = {**_identity(item), "result": PENDING}
                    self._repo.save_ledger(ledger)
                outcome = self._fetch(item, state)
                if outcome is None:
                    report.stop_reason = "rate_limited"
                    break
                self._record_result(ledger, item, outcome)
                fetch_report.outcomes.append(outcome)
                (report.done_now if outcome.state in DONE_STATES else report.failed_now).append(
                    outcome
                )
        finally:
            self._quota.save(state.to_dict())
            self._repo.write_summary(fetch_report)
        return self._finish(report, items, state, limits, ledger)

    # ---- 请求记录（T10-M0b 第 6 节） ----

    def is_done(self, ledger: dict[str, Any], item: BatchItem) -> bool:
        """统一的完成判定（MR-01）：身份一致，并且记录为 ok 且两个文件的哈希都相符，
        或者记录为 no_data 且没有矛盾文件。用于跳过、cumulative_done 与 remaining。"""
        rec = ledger["items"].get(item.symbol)
        if rec is None or not _same_identity(rec, item):
            return False
        hashes = self._repo.file_hashes(item.symbol)
        if rec.get("result") == "ok":
            return hashes == rec.get("sha256")
        if rec.get("result") == "no_data":
            return hashes["price"] is None and hashes["actions"] is None
        return False

    def _record_result(
        self, ledger: dict[str, Any], item: BatchItem, outcome: FetchOutcome
    ) -> None:
        rec = {
            **_identity(item),
            "result": outcome.state if outcome.state in DONE_STATES else FAILED,
        }
        if outcome.state == "ok":
            rec["sha256"] = self._repo.file_hashes(item.symbol)
        ledger["items"][item.symbol] = rec
        self._repo.save_ledger(ledger)

    def _verify_directory(self, items: Sequence[BatchItem]) -> dict[str, Any]:
        """运行开始时的判定（任务卡第 6 节第 2 项的表）。通过时返回请求记录（新目录为空记录）。"""
        ledger = self._repo.load_ledger()
        problems: list[str] = []
        if ledger is None:
            if self._repo.has_legacy_files():
                raise UnverifiedDirectory(
                    "旧目录没有请求记录（_request_ledger.json）：已有状态或数据文件来历不明，"
                    "不能复用。建议改用新目录。"
                )
            ledger = {"source": self._source.name, "items": {}}
        elif ledger.get("source") != self._source.name:
            problems.append(
                f"数据源名称不同：记录为 {ledger.get('source')!r}，本次为 {self._source.name!r}"
            )
        for item in items:
            problems += self._check_item(ledger, item)
        if problems:
            raise UnverifiedDirectory(
                "目标目录不能复用（未发出任何请求，未改动任何文件）：\n  "
                + "\n  ".join(problems)
                + "\n建议改用新目录。"
            )
        return ledger

    def _check_item(self, ledger: dict[str, Any], item: BatchItem) -> list[str]:
        sym = item.symbol
        rec = ledger["items"].get(sym)
        hashes = self._repo.file_hashes(sym)
        files = [k for k in ("price", "actions") if hashes[k] is not None]
        if rec is None:
            if files or self._repo.has_status(sym):
                return [f"{sym}：不在请求记录中，但目录中已有它的文件或状态（未经验证的旧数据）"]
            return []
        diff = [k for k, v in _identity(item).items() if rec.get(k) != v]
        if diff:
            return [
                f"{sym}：{k} 不一致（记录 {rec.get(k)!r}，本次 {_identity(item)[k]!r}）"
                for k in diff
            ]
        result = rec.get("result")
        if result == "ok":
            out = []
            for k, label in (("price", f"{sym}.csv"), ("actions", f"{sym}.actions.csv")):
                if hashes[k] is None:
                    out.append(f"{sym}：记录为 ok，但 {label} 缺失")
                elif hashes[k] != (rec.get("sha256") or {}).get(k):
                    out.append(f"{sym}：记录为 ok，但 {label} 的 SHA-256 与记录不符")
            return out
        if result == "no_data":
            return [f"{sym}：记录为 no_data，但目录中有 {k} 文件（与无数据矛盾）" for k in files]
        if files:  # pending 或 failed
            return [f"{sym}：记录为 {result}，但目录中有它的 {k} 文件（来历不明）" for k in files]
        return []

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
        before = self._source.request_count
        try:
            raw = self._source.fetch(item.source_ticker, item.start, item.end)
        except RateLimitError:
            self._progress("    连续 3 次 429：额度已用尽，稍后再运行")
            return None
        except PriceSourceError as e:
            return record_failure(self._repo, member, str(e), self._progress)
        finally:
            sent = self._source.request_count - before
            state.day_requests += sent
            if sent > 0:  # Gemini 第二阶段 B-04：只有确实发出请求时才计入当月新代码
                state.add_ticker(item.source_ticker)
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
        ledger: dict[str, Any],
    ) -> BatchReport:
        report.cumulative_done = sum(1 for i in items if self.is_done(ledger, i))
        report.day_requests = state.day_requests
        report.month_unique = len(state.month_tickers)
        per_run = min(limits.per_day, limits.unique_per_month, limits.max_symbols or limits.per_day)
        report.estimated_runs = math.ceil(report.remaining / per_run) if report.remaining else 0
        return report


def _identity(item: BatchItem) -> dict[str, str]:
    return {
        "source_ticker": item.source_ticker,
        "start": item.start.isoformat(),
        "end": item.end.isoformat(),
    }


def _same_identity(rec: dict[str, Any], item: BatchItem) -> bool:
    return all(rec.get(k) == v for k, v in _identity(item).items())
