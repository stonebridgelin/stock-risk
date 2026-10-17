"""用例：中转服务接入核验 `verify-massive-relay`（T08d-2c 第 2、3 节）。只核验，不入库。

依次执行（每完成一项保存进度，中断后续传）：
1. 第 2 节接口覆盖试探：日线（MSFT，2019-01，未复权与复权各一次）、拆股（AAPL）、分红（KO）、
   证券清单（2019-06-03，market=stocks，active 真 / 假，各只取第一页）、证券详情（MSFT）；
   记录状态码、返回字段名与条数，不记录密钥。404 或"不支持"的记为不可用，第 3 节依赖它的部分跳过；
2. 第 3.1 节价格对账：27 只探索股的未复权日线（2010-01-01 至 2019-12-31）与 Tiingo 逐日对账；
3. 第 3.2 节参考数据：按 T08d-2a 的 32 种清单查询从中转服务取一份，与官方快照逐条比对。
任何请求返回 401 或 403：记录原文后立即停止全部运行，不换密钥重试；连续 3 次 429 保存进度后退出。
封存：请求的 to / date 不晚于 2019-12-31；响应中晚于截止日的行在内存中丢弃并计数。
照实记录，不作判定（判定由产品经理按任务卡第 4 节执行）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from stock_risk.application.ports import PriceSourceError, QueryRejected, RateLimitError
from stock_risk.application.progress import ProgressFn, quiet
from stock_risk.domain.massive_followup import TickerPages
from stock_risk.domain.massive_labels import seal_list_row
from stock_risk.domain.massive_paid import ReferenceBar, seal_bars, seal_dividend, seal_split
from stock_risk.domain.massive_probe import ListQuery, list_queries, seal_ticker
from stock_risk.domain.massive_verify import VERIFY_FROM, VERIFY_TO, compare_daily, compare_snapshot
from stock_risk.domain.seal import EXPLORATION_END

AUTH_REJECT_STATUSES = (401, 403)
LIST_MAX_PAGES = 200
COVERAGE_DAY = date(2019, 6, 3)
# 第 3 节依赖的接口：价格对账依赖未复权日线，参考数据依赖两种证券清单
PRICE_NEEDS = ("aggs_raw",)
REFERENCE_NEEDS = ("list_active", "list_inactive")


class RelaySource(Protocol):
    """中转服务（只读）。原始记录只在内存中返回。"""

    @property
    def request_count(self) -> int: ...

    @property
    def source_host(self) -> str: ...

    def aggs(self, ticker: str, start: date, end: date, adjusted: bool) -> TickerPages | None: ...

    def splits(self, ticker: str, until: date) -> TickerPages: ...

    def dividends(self, ticker: str, until: date) -> TickerPages: ...

    def list_tickers_pages(self, query: ListQuery, max_pages: int) -> TickerPages: ...

    def overview(self, ticker: str, day: date) -> dict[str, Any] | None: ...


class VerifyStore(Protocol):
    def load_progress(self) -> dict[str, Any]: ...

    def save_progress(self, progress: Mapping[str, Any]) -> None: ...

    def save_price_rows(self, ticker: str, rows: Sequence[Mapping[str, Any]]) -> None: ...

    def save_relay_list(self, key: str, rows: Sequence[Mapping[str, Any]]) -> None: ...

    def load_relay_list(self, key: str) -> list[dict[str, str]]: ...

    def write_outputs(
        self,
        coverage: Sequence[Mapping[str, Any]],
        prices: Sequence[Mapping[str, Any]],
        reference: Sequence[Mapping[str, Any]],
        examples: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class Probe:
    """第 2 节的一次最小请求。"""

    name: str
    request: str
    call: Callable[[RelaySource], Any]
    seal: Callable[[list[dict[str, Any]]], tuple[int, int]] | None = None  # → (保留, 丢弃)


def _seal_rows(fn: Callable[[Mapping[str, Any]], Any]) -> Callable[[list[dict]], tuple[int, int]]:  # type: ignore[type-arg]
    def count(rows: list[dict]) -> tuple[int, int]:  # type: ignore[type-arg]
        kept = sum(1 for r in rows if fn(r) is not None)
        return kept, len(rows) - kept

    return count


def _seal_aggs(rows: list[dict]) -> tuple[int, int]:  # type: ignore[type-arg]
    bars, dropped = seal_bars(rows)
    return len(bars), dropped


def coverage_probes() -> tuple[Probe, ...]:
    """第 2 节的 7 次请求（任务卡表格的 6 类；日线两次）。"""
    jan = (date(2019, 1, 2), date(2019, 1, 31))
    stocks = "market=stocks，date=2019-06-03，只取第一页"
    return (
        Probe("aggs_raw", "日线 MSFT 2019-01-02—2019-01-31 adjusted=false",
              lambda s: s.aggs("MSFT", *jan, False), _seal_aggs),
        Probe("aggs_adj", "日线 MSFT 2019-01-02—2019-01-31 adjusted=true",
              lambda s: s.aggs("MSFT", *jan, True), _seal_aggs),
        Probe("splits", "拆股 AAPL execution_date ≤ 2019-12-31",
              lambda s: s.splits("AAPL", EXPLORATION_END), _seal_rows(seal_split)),
        Probe("dividends", "分红 KO ex_dividend_date ≤ 2019-12-31",
              lambda s: s.dividends("KO", EXPLORATION_END), _seal_rows(seal_dividend)),
        Probe("list_active", f"证券清单 active=true，{stocks}",
              lambda s: s.list_tickers_pages(_stocks(True), 1), _seal_rows(seal_list_row)),
        Probe("list_inactive", f"证券清单 active=false，{stocks}",
              lambda s: s.list_tickers_pages(_stocks(False), 1), _seal_rows(seal_list_row)),
        Probe("overview", "证券详情 MSFT date=2019-06-03",
              lambda s: s.overview("MSFT", COVERAGE_DAY)),
    )  # fmt: skip


def _stocks(active: bool) -> ListQuery:
    return ListQuery(COVERAGE_DAY, None, None, active, market="stocks")


@dataclass(frozen=True, slots=True)
class _Fail:
    """被拒绝或出错（照实记录的状态）。"""

    status: dict[str, Any]


class _Stop(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(slots=True)
class VerifyResult:
    summary: dict[str, Any]
    paths: list[str]
    stopped: str | None = None  # auth_rejected / rate_limited
    requests_now: int = 0
    remaining: list[str] = field(default_factory=list)


def _fields(rows: Sequence[Mapping[str, Any]]) -> str:
    return "|".join(sorted({k for r in rows for k in r}))


class VerifyMassiveRelayService:
    def __init__(
        self,
        source: RelaySource,
        store: VerifyStore,
        calendar: Sequence[date],
        tickers: Sequence[str],
        tiingo_prices: Mapping[str, Sequence[ReferenceBar]],
        official_lists: Mapping[str, Sequence[Mapping[str, Any]]],
        clock: Callable[[], float],
        progress: ProgressFn = quiet,
        queries: Sequence[ListQuery] | None = None,
    ) -> None:
        self._source = source
        self._store = store
        self._calendar = tuple(calendar)
        self._tickers = tuple(tickers)
        self._tiingo = tiingo_prices
        self._official = official_lists
        self._clock = clock
        self._progress = progress
        self._queries = tuple(queries) if queries is not None else list_queries()
        self._probes = coverage_probes()

    @property
    def source_host(self) -> str:
        return self._source.source_host

    @property
    def planned(self) -> dict[str, int]:
        """计划的请求项数（清单查询可能翻页，实际请求数可能更多）。"""
        lists = sum(1 for q in self._queries if q.key in self._official)
        return {"coverage": len(self._probes), "prices": len(self._tickers), "lists": lists}

    def run(self) -> VerifyResult:
        state = self._store.load_progress()
        for k in ("coverage", "prices", "lists"):
            state.setdefault(k, {})
        state.setdefault("requests", 0)
        state.setdefault("elapsed_seconds", 0.0)
        before, t0 = self._source.request_count, self._clock()
        stopped: str | None = state.get("auth_rejected") and "auth_rejected"
        try:
            if stopped:
                raise _Stop(stopped)
            self._run_coverage(state)
            self._run_prices(state)
            self._run_lists(state)
        except _Stop as e:
            stopped = e.reason
            self._progress(f"    停止：{e.reason}，进度已保存")
        finally:
            state["requests"] += self._source.request_count - before
            state["elapsed_seconds"] = round(state["elapsed_seconds"] + self._clock() - t0, 1)
            self._store.save_progress(state)
        coverage, prices = self._coverage_rows(state), self._price_rows(state)
        reference, examples = self._reference(state)
        summary = self._summary(state, stopped)
        paths = self._store.write_outputs(coverage, prices, reference, examples, summary)
        return VerifyResult(
            summary, paths, stopped, self._source.request_count - before, self._remaining(state)
        )

    # ---- 请求 ----

    def _call(self, state: dict[str, Any], what: str, fn: Callable[[], Any]) -> Any:
        """发出请求。401 / 403 记录后立即停止全部运行；429 用尽停止；其余拒绝或错误返回状态字典。"""
        try:
            return fn()
        except RateLimitError:
            raise _Stop("rate_limited") from None
        except QueryRejected as e:
            self._progress(f"    被拒绝：HTTP {e.http_status} {e.message}")
            status = {"status": "rejected", "http_status": e.http_status, "message": e.message}
            if e.http_status in AUTH_REJECT_STATUSES:
                state["auth_rejected"] = {"what": what, **status}
                self._store.save_progress(state)
                raise _Stop("auth_rejected") from None
            return _Fail(status)
        except PriceSourceError as e:
            self._progress(f"    失败：{e}")
            return _Fail({"status": "error", "message": str(e)})

    def _available(self, state: Mapping[str, Any], names: Sequence[str]) -> bool:
        return all(state["coverage"].get(n, {}).get("status") == "ok" for n in names)

    # ---- 第 2 节 ----

    def _run_coverage(self, state: dict[str, Any]) -> None:
        for p in self._probes:
            if p.name in state["coverage"]:
                continue
            self._progress(f"[试探] {p.request}")
            got = self._call(state, p.name, lambda p=p: p.call(self._source))
            state["coverage"][p.name] = _coverage_status(p, got)
            self._store.save_progress(state)

    # ---- 第 3.1 节 ----

    def _run_prices(self, state: dict[str, Any]) -> None:
        if not self._available(state, PRICE_NEEDS):
            return
        for t in self._tickers:
            if t in state["prices"]:
                continue
            self._progress(f"[价格] {t}")
            got = self._call(
                state,
                f"prices_{t}",
                lambda t=t: self._source.aggs(t, VERIFY_FROM, VERIFY_TO, False),
            )
            if isinstance(got, _Fail):
                state["prices"][t] = got.status
            elif got is None:
                state["prices"][t] = {"status": "not_found"}
            elif not got.complete:
                state["prices"][t] = {"status": "error", "message": f"分页超过 {got.pages} 页"}
            elif t not in self._tiingo:
                state["prices"][t] = {"status": "no_tiingo"}
            else:
                bars, dropped = seal_bars(got.rows)
                rows, summary = compare_daily(bars, self._tiingo[t], self._calendar)
                self._store.save_price_rows(t, rows)
                state["prices"][t] = {"status": "ok", "dropped_after_2019": dropped, **summary}
                self._progress(f"    共同日 {summary['common_days']}，丢弃 {dropped} 行")
            self._store.save_progress(state)

    # ---- 第 3.2 节 ----

    def _run_lists(self, state: dict[str, Any]) -> None:
        if not self._available(state, REFERENCE_NEEDS):
            return
        for q in self._queries:
            if q.key in state["lists"] or q.key not in self._official:
                continue
            self._progress(f"[清单] {q.key}")
            got = self._call(
                state,
                f"list_{q.key}",
                lambda q=q: self._source.list_tickers_pages(q, LIST_MAX_PAGES),
            )
            if isinstance(got, _Fail):
                state["lists"][q.key] = got.status
            elif not got.complete:
                state["lists"][q.key] = {"status": "error", "message": f"分页超过 {got.pages} 页"}
            else:
                sealed = [s for s in (seal_ticker(r) for r in got.rows) if s is not None]
                self._store.save_relay_list(q.key, [s.as_row() for s in sealed])
                state["lists"][q.key] = {
                    "status": "ok",
                    "pages": got.pages,
                    "rows": len(sealed),
                    "dropped_list_date_after_2019": len(got.rows) - len(sealed),
                    "delisted_after_2019": sum(s.delisted_after_2019 for s in sealed),
                }
            self._store.save_progress(state)

    # ---- 汇总 ----

    def _coverage_rows(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "name": p.name,
                "request": p.request,
                **state["coverage"].get(p.name, {"status": "not_run"}),
            }
            for p in self._probes
        ]

    def _price_rows(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        reason = None if self._available(state, PRICE_NEEDS) else "skipped_unavailable"
        return [
            {"ticker": t, **state["prices"].get(t, {"status": reason or "not_run"})}
            for t in self._tickers
        ]

    def _reference(
        self, state: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        unavailable = not self._available(state, REFERENCE_NEEDS)
        summaries: list[dict[str, Any]] = []
        examples: list[dict[str, Any]] = []
        for day in sorted({q.day for q in self._queries if q.key in self._official}):
            qs = [q for q in self._queries if q.day == day and q.key in self._official]
            statuses = [state["lists"].get(q.key, {}).get("status") for q in qs]
            if unavailable or any(s != "ok" for s in statuses):
                status = "skipped_unavailable" if unavailable else "incomplete"
                failed = [q.key for q, s in zip(qs, statuses, strict=True) if s != "ok"]
                summaries.append(
                    {"date": day.isoformat(), "status": status, "message": "|".join(failed)}
                )
                continue
            official = {q.key: self._official[q.key] for q in qs}
            relay = {q.key: self._store.load_relay_list(q.key) for q in qs}
            row, ex = compare_snapshot(day, official, relay)
            summaries.append({"status": "ok", **row})
            examples.extend(ex)
        return summaries, examples

    def _seal_counts(self, state: Mapping[str, Any]) -> dict[str, int]:
        cov = state["coverage"].values()
        prices = [v for v in state["prices"].values() if v.get("status") == "ok"]
        lists = [v for v in state["lists"].values() if v.get("status") == "ok"]
        return {
            "coverage_rows_dropped_after_2019": sum(v.get("dropped_after_2019") or 0 for v in cov),
            "price_rows_dropped_after_2019": sum(v["dropped_after_2019"] for v in prices),
            "list_rows_dropped_list_date_after_2019": sum(
                v["dropped_list_date_after_2019"] for v in lists
            ),
            "list_rows_delisted_after_2019_masked": sum(v["delisted_after_2019"] for v in lists),
        }

    def _summary(self, state: Mapping[str, Any], stopped: str | None) -> dict[str, Any]:
        return {
            "stopped": stopped,
            "auth_rejected": state.get("auth_rejected"),
            "unavailable": [
                p.name for p in self._probes if state["coverage"].get(p.name, {}).get("status")
                not in (None, "ok")
            ],
            "seal": self._seal_counts(state),
            "requests_total": state["requests"],
            "elapsed_seconds_total": state["elapsed_seconds"],
        }  # fmt: skip

    def _remaining(self, state: Mapping[str, Any]) -> list[str]:
        out = [p.name for p in self._probes if p.name not in state["coverage"]]
        pending = bool(out)  # 试探未完成时，第 3 节各项都还未定
        if pending or self._available(state, PRICE_NEEDS):
            out += [f"prices_{t}" for t in self._tickers if t not in state["prices"]]
        if pending or self._available(state, REFERENCE_NEEDS):
            out += [
                f"list_{q.key}"
                for q in self._queries
                if q.key in self._official and q.key not in state["lists"]
            ]
        return out


def _coverage_status(p: Probe, got: Any) -> dict[str, Any]:
    """一次试探的结果：状态码、页数、条数、封存丢弃数、返回字段名。"""
    if isinstance(got, _Fail):
        return got.status
    if got is None:
        return {"status": "not_found", "http_status": 404}
    if isinstance(got, TickerPages):
        kept, dropped = p.seal(got.rows) if p.seal else (len(got.rows), 0)
        return {
            "status": "ok",
            "http_status": 200,
            "pages": got.pages,
            "rows": kept,
            "dropped_after_2019": dropped,
            "result_fields": _fields(got.rows),
        }
    return {"status": "ok", "http_status": 200, "rows": 1, "result_fields": _fields([got])}
