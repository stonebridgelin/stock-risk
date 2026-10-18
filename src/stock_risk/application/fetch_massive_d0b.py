"""D0b 补取（T12-D0b 第 2.3 节）：只请求缺口清单中的请求，按请求身份记账、断点续传。

- 请求身份 = (主机, 类别, 源代码, 请求起止日, adjusted)（DB-02）。日线带起止日与 adjusted；
  拆股、分红的旧路径 /v3/reference/splits、dividends 只有上限
  （执行日或除息日不晚于请求止日），下限记为空。
- "完成" = 成功且分页取完，且输出文件的哈希与账本一致；跳过、已完成数、剩余数都用这一个判定。
- 封存：请求止日晚于 2019-12-31 的，在计划阶段拒绝（日线另有 check_price_range），不发出任何请求；
  响应中的越界行由存储层在写文件之前过滤并计数。
- 401、403 立即停止；连续 429 保存账本后停止；其他错误记失败，继续下一项。
- 运行前的确认（输入 yes）与 --dry-run 由命令行负责；本服务的 plan() 不联网。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from stock_risk.application.ports import PriceSourceError, QueryRejected, RateLimitError
from stock_risk.application.progress import ProgressFn, quiet
from stock_risk.domain.massive_followup import TickerPages
from stock_risk.domain.massive_paid import check_price_range
from stock_risk.domain.seal import EXPLORATION_END, SealViolation

RAW, ADJ, SPLITS, DIVIDENDS = "raw", "adj", "splits", "dividends"
CATEGORIES = (RAW, ADJ, SPLITS, DIVIDENDS)
OK, FAILED, INCOMPLETE = "ok", "failed", "incomplete"
RELAY_HOST = "relaynex.vip"  # T08d-2c 判定通过的主机；D0b 只经它取数


@dataclass(frozen=True, slots=True)
class RequestSpec:
    """缺口清单的一行。start、end 为该段的请求范围（2.1 的 range 与分段的交集）。"""

    symbol: str
    ticker: str
    category: str
    start: date
    end: date

    @property
    def adjusted(self) -> bool | None:
        return {RAW: False, ADJ: True}.get(self.category)

    @property
    def request_start(self) -> date | None:
        return self.start if self.category in (RAW, ADJ) else None

    def key(self, host: str) -> str:
        adj = {True: "true", False: "false", None: ""}[self.adjusted]
        s = self.request_start.isoformat() if self.request_start else ""
        return f"{host}|{self.category}|{self.ticker}|{s}|{self.end.isoformat()}|{adj}"


@dataclass(frozen=True, slots=True)
class SavedFile:
    path: str  # 相对下载目录
    sha256: str
    rows: int
    sealed_dropped: int
    masked: int  # 分红：其他日期字段被遮蔽的行数
    currency_missing: int  # 分红：响应中没有币种字段的行数


class D0bSource(Protocol):
    @property
    def source_host(self) -> str: ...

    @property
    def request_count(self) -> int: ...

    def aggs(self, ticker: str, start: date, end: date, adjusted: bool) -> TickerPages | None: ...

    def splits(self, ticker: str, until: date) -> TickerPages: ...

    def dividends(self, ticker: str, until: date) -> TickerPages: ...


class D0bStore(Protocol):
    def load_ledger(self) -> dict[str, Any]: ...

    def save_ledger(self, ledger: Mapping[str, Any]) -> None: ...

    def save(
        self, category: str, ticker: str, results: Sequence[Mapping[str, Any]]
    ) -> SavedFile: ...

    def file_sha256(self, rel: str) -> str | None: ...


def check_specs(specs: Sequence[RequestSpec]) -> None:
    """计划阶段的检查：类别合法、范围合法、止日不晚于 2019-12-31（拒绝发出越界请求）。"""
    seen: set[tuple[str, str, date, date]] = set()
    for s in specs:
        if s.category not in CATEGORIES:
            raise ValueError(f"未知的请求类别: {s.category}")
        if s.end > EXPLORATION_END:
            raise SealViolation(
                f"{s.ticker} {s.category} 请求止日 {s.end} 晚于 {EXPLORATION_END}，拒绝"
            )
        if s.category in (RAW, ADJ):
            check_price_range(s.start, s.end)
        elif s.start > s.end:
            raise ValueError(f"{s.ticker} {s.category}: 起日晚于止日")
        ident = (s.ticker, s.category, s.start, s.end)
        if ident in seen:
            raise ValueError(f"缺口清单重复: {ident}")
        seen.add(ident)


@dataclass(frozen=True, slots=True)
class Plan:
    host: str
    pending: tuple[RequestSpec, ...]
    done: tuple[RequestSpec, ...]

    def counts(self) -> dict[str, int]:
        out = {c: 0 for c in CATEGORIES}
        for s in self.pending:
            out[s.category] += 1
        return out


@dataclass
class FetchReport:
    host: str
    requests_now: int = 0
    completed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    incomplete: list[str] = field(default_factory=list)
    stopped: str | None = None  # 401 / 403 / 限频
    remaining: int = 0


def is_complete(entry: Mapping[str, Any] | None, store: D0bStore) -> bool:
    if not entry or entry.get("status") != OK or not entry.get("complete"):
        return False
    return store.file_sha256(str(entry.get("file"))) == entry.get("sha256")


class FetchMassiveD0bService:
    def __init__(
        self,
        specs: Sequence[RequestSpec],
        host: str,
        store_factory: Callable[[], D0bStore],
        source_factory: Callable[[], D0bSource] | None = None,
        progress: ProgressFn = quiet,
    ) -> None:
        check_specs(specs)
        if host != RELAY_HOST:
            raise ValueError(f"D0b 补取只经中转服务 {RELAY_HOST}，不连接 {host}")
        self._specs = tuple(specs)
        self._host = host
        self._store_factory = store_factory
        self._source_factory = source_factory
        self._progress = progress

    def plan(self) -> Plan:
        """不联网：按账本区分已完成与待请求。"""
        store = self._store_factory()
        ledger = store.load_ledger().get("requests", {})
        done = [s for s in self._specs if is_complete(ledger.get(s.key(self._host)), store)]
        pending = [s for s in self._specs if s not in done]
        return Plan(self._host, tuple(pending), tuple(done))

    def run(self) -> FetchReport:
        if self._source_factory is None:
            raise RuntimeError("没有数据源：--dry-run 只能调用 plan()")
        store = self._store_factory()
        source = self._source_factory()
        if source.source_host != self._host:
            raise ValueError(
                f"数据源主机 {source.source_host} 与计划主机 {self._host} 不一致，停止"
            )
        ledger = store.load_ledger()
        requests: dict[str, Any] = ledger.setdefault("requests", {})
        report = FetchReport(self._host)
        before = source.request_count
        pending = [
            s for s in self._specs if not is_complete(requests.get(s.key(self._host)), store)
        ]
        for i, spec in enumerate(pending):
            key = spec.key(self._host)
            entry: dict[str, Any] = {
                "symbol": spec.symbol,
                "category": spec.category,
                "ticker": spec.ticker,
                "start": spec.request_start.isoformat() if spec.request_start else None,
                "end": spec.end.isoformat(),
                "adjusted": spec.adjusted,
            }
            try:
                got = self._call(source, spec)
            except QueryRejected as e:
                entry |= {"status": FAILED, "complete": False, "http_status": e.http_status}
                entry["message"] = str(e)
                requests[key] = entry
                store.save_ledger(ledger)
                report.failed.append(key)
                if e.http_status in (401, 403):
                    report.stopped = f"HTTP {e.http_status}"
                    report.remaining = len(pending) - i - 1
                    break
                continue
            except RateLimitError:
                report.stopped = "连续 429"
                report.remaining = len(pending) - i
                break
            except PriceSourceError as e:
                entry |= {"status": FAILED, "complete": False, "message": str(e)}
                requests[key] = entry
                store.save_ledger(ledger)
                report.failed.append(key)
                continue
            if got is None:
                entry |= {"status": FAILED, "complete": False, "http_status": 404}
                entry["message"] = "not found"
                report.failed.append(key)
            elif not got.complete:
                entry |= {"status": INCOMPLETE, "complete": False, "pages": got.pages}
                report.incomplete.append(key)
            else:
                saved = store.save(spec.category, spec.ticker, got.rows)
                entry |= {
                    "status": OK,
                    "complete": True,
                    "pages": got.pages,
                    "rows": saved.rows,
                    "sealed_dropped": saved.sealed_dropped,
                    "file": saved.path,
                    "sha256": saved.sha256,
                }
                if spec.category == DIVIDENDS:
                    entry["dates_masked"] = saved.masked
                    entry["currency_field_missing"] = saved.currency_missing
                report.completed.append(key)
            requests[key] = entry
            store.save_ledger(ledger)
            self._progress(
                f"[{i + 1}/{len(pending)}] {spec.ticker} {spec.category}：{entry['status']}"
            )
        report.requests_now = source.request_count - before
        return report

    @staticmethod
    def _call(source: D0bSource, spec: RequestSpec) -> TickerPages | None:
        if spec.category in (RAW, ADJ):
            return source.aggs(spec.ticker, spec.start, spec.end, spec.category == ADJ)
        if spec.category == SPLITS:
            return source.splits(spec.ticker, spec.end)
        return source.dividends(spec.ticker, spec.end)
