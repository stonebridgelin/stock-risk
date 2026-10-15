"""用例：Massive 追加探查 `probe-massive --followup`（T08d-2a 第 2.4 节）。不下载价格。

- A：去掉类型过滤的清单（16 种组合），按 type 计数；
- B：去掉交易所与类型过滤（market=stocks、active=true，4 个日期），
  按 primary_exchange × type 交叉计数，只保存计数，不保存逐行内容；
  第一个日期翻页超过 40 页时停止 B，记录实际页数与已得到的计数；
- C：补充案例（Ticker Overview 加 date）；
- D：与 T08d-1 的 Tiingo 探查按代码对照（不发请求）。
封存处理、断点续传、被拒绝照实记录与 429 退出同 probe_massive。照实记录，不作结论。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from stock_risk.application.ports import PriceSourceError, QueryRejected, RateLimitError
from stock_risk.application.probe_massive import case_result, case_rows, sealed_from_row
from stock_risk.application.progress import ProgressFn, quiet
from stock_risk.domain.massive_followup import (
    B_FIRST_DATE_MAX_PAGES,
    FOLLOWUP_CASES,
    ReferenceSpan,
    TickerPages,
    coverage_compare,
    crosstab,
    followup_market_queries,
    followup_type_queries,
    massive_on,
    tiingo_on,
    type_profile,
)
from stock_risk.domain.massive_probe import Case, ListQuery, seal_all

FULL_MAX_PAGES = 200  # A 与 B 其余日期的翻页上限（与适配器 MAX_PAGES 一致，防止 next_url 循环）


class PagedReferenceSource(Protocol):
    """Massive 参考数据（只读），可限定翻页数。原始记录只在内存中返回。"""

    @property
    def request_count(self) -> int: ...

    def list_tickers_pages(self, query: ListQuery, max_pages: int) -> TickerPages: ...

    def overview(self, ticker: str, day: date) -> dict[str, Any] | None: ...


class FollowupStore(Protocol):
    """进度与输出。只接收封存处理后的行与计数。"""

    def load_progress(self) -> dict[str, Any]: ...

    def save_progress(self, progress: Mapping[str, Any]) -> None: ...

    def write_outputs(
        self,
        lists: Mapping[date, Sequence[Mapping[str, Any]]],
        crosstabs: Mapping[date, Mapping[str, Mapping[str, int]]],
        cases: Sequence[Mapping[str, Any]],
        coverage: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> list[str]: ...


@dataclass(slots=True)
class FollowupResult:
    summary: dict[str, Any]
    paths: list[str]
    rate_limited: bool = False
    requests_now: int = 0
    remaining: list[str] = field(default_factory=list)


def _status(kind: str, **extra: Any) -> dict[str, Any]:
    return {"status": kind, **extra}


class _Stop(Exception):
    """连续 3 次 429：保存进度后退出。"""


class ProbeMassiveFollowupService:
    def __init__(
        self,
        source: PagedReferenceSource,
        store: FollowupStore,
        tiingo: Sequence[ReferenceSpan],
        clock: Callable[[], float],
        progress: ProgressFn = quiet,
        type_queries: Sequence[ListQuery] | None = None,
        market_queries: Sequence[ListQuery] | None = None,
        cases: Sequence[Case] = FOLLOWUP_CASES,
        b_first_max_pages: int = B_FIRST_DATE_MAX_PAGES,
    ) -> None:
        self._source = source
        self._store = store
        self._tiingo = tuple(tiingo)
        self._clock = clock
        self._progress = progress
        self._a = tuple(type_queries) if type_queries is not None else followup_type_queries()
        self._b = tuple(market_queries) if market_queries is not None else followup_market_queries()
        self._cases = tuple(cases)
        self._b_first_max = b_first_max_pages
        self._mark = (0, 0.0)  # 上次保存进度时的（请求数, 时钟）

    def run(self) -> FollowupResult:
        state = self._store.load_progress()
        for k in ("a", "b", "cases"):
            state.setdefault(k, {})
        state.setdefault("requests", 0)
        state.setdefault("elapsed_seconds", 0.0)
        before = self._source.request_count
        self._mark = (before, self._clock())
        rate_limited = False
        try:
            self._run_a(state)
            self._run_b(state)
            self._run_cases(state)
        except _Stop:
            self._progress("    连续 3 次 429：保存进度后退出，稍后再运行")
            rate_limited = True
        finally:
            self._save(state)
        summary, coverage = self._summary(state)
        paths = self._store.write_outputs(
            self._lists(state),
            self._crosstabs(state),
            case_rows(self._cases, state["cases"]),
            coverage,
            summary,
        )
        remaining = [q.key for q in self._a if q.key not in state["a"]]
        remaining += [q.key for q in self._b if q.key not in state["b"]]
        remaining += [c.key for c in self._cases if c.key not in state["cases"]]
        return FollowupResult(
            summary, paths, rate_limited, self._source.request_count - before, remaining
        )

    def _save(self, state: dict[str, Any]) -> None:
        """保存进度；请求数与耗时每次保存时累计，进程被中断也不丢失已发生的计数。"""
        n, t = self._source.request_count, self._clock()
        state["requests"] += n - self._mark[0]
        state["elapsed_seconds"] = round(state["elapsed_seconds"] + t - self._mark[1], 1)
        self._mark = (n, t)
        self._store.save_progress(state)

    # ---- 请求 ----

    def _fetch(self, q: ListQuery, max_pages: int) -> TickerPages | dict[str, Any]:
        """翻页查询；被拒绝或出错时返回状态字典，429 用尽抛 _Stop。"""
        try:
            return self._source.list_tickers_pages(q, max_pages)
        except RateLimitError:
            raise _Stop from None
        except QueryRejected as e:
            self._progress(f"    被拒绝：HTTP {e.http_status} {e.message}")
            return _status("rejected", http_status=e.http_status, message=e.message)
        except PriceSourceError as e:
            self._progress(f"    失败：{e}")
            return _status("error", message=str(e))

    def _run_a(self, state: dict[str, Any]) -> None:
        for q in self._a:
            if q.key in state["a"]:
                continue
            self._progress(f"[A] {q.key}")
            got = self._fetch(q, FULL_MAX_PAGES)
            if isinstance(got, dict):
                state["a"][q.key] = got
            elif not got.complete:
                state["a"][q.key] = _status("error", message=f"分页超过 {got.pages} 页，停止")
            else:
                rows = seal_all(got.rows)
                state["a"][q.key] = _status(
                    "ok",
                    pages=got.pages,
                    profile=type_profile(rows, inactive=not q.active),
                    rows=[r.as_row() for r in rows],
                )
                self._progress(f"    {len(rows)} 条，{got.pages} 页")
            self._save(state)

    def _run_b(self, state: dict[str, Any]) -> None:
        for i, q in enumerate(self._b):
            if q.key in state["b"]:
                continue
            if state.get("b_stopped"):
                state["b"][q.key] = _status(
                    "skipped", message="B 的第一个日期翻页超过上限，B 已停止"
                )
                self._save(state)
                continue
            first = i == 0
            cap = self._b_first_max if first else FULL_MAX_PAGES
            self._progress(f"[B] {q.key}")
            got = self._fetch(q, cap)
            if isinstance(got, dict):
                state["b"][q.key] = got
            else:
                rows = seal_all(got.rows)  # 只在内存中计数，不保存逐行内容
                kind = "ok" if got.complete else "stopped_at_page_limit"
                state["b"][q.key] = _status(
                    kind, pages=got.pages, count=len(rows), crosstab=crosstab(rows)
                )
                self._progress(f"    {len(rows)} 条，{got.pages} 页（{kind}）")
                if not got.complete and first:
                    state["b_stopped"] = True
            self._save(state)

    def _run_cases(self, state: dict[str, Any]) -> None:
        for c in self._cases:
            if c.key in state["cases"]:
                continue
            self._progress(f"[C] {c.ticker} @ {c.day}")
            try:
                raw = self._source.overview(c.ticker, c.day)
            except RateLimitError:
                raise _Stop from None
            except QueryRejected as e:
                state["cases"][c.key] = _status(
                    "rejected", http_status=e.http_status, message=e.message
                )
            except PriceSourceError as e:
                state["cases"][c.key] = _status("error", message=str(e))
            else:
                state["cases"][c.key] = case_result(raw)
            self._progress(f"    {state['cases'][c.key]['status']}")
            self._save(state)

    # ---- 输出 ----

    def _lists(self, state: Mapping[str, Any]) -> dict[date, list[dict[str, Any]]]:
        out: dict[date, list[dict[str, Any]]] = {}
        for q in self._a:
            st = state["a"].get(q.key)
            if st is None or st["status"] != "ok":
                continue
            query = {"query_exchange": q.exchange, "query_active": q.active}
            out.setdefault(q.day, []).extend({**query, **r} for r in st["rows"])
        return out

    def _crosstabs(self, state: Mapping[str, Any]) -> dict[date, dict[str, dict[str, int]]]:
        out = {}
        for q in self._b:
            st = state["b"].get(q.key)
            if st is not None and "crosstab" in st:
                out[q.day] = st["crosstab"]
        return out

    def _coverage(self, state: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """D：A 中 active=true 的两个交易所都成功的日期才对照，否则记 incomplete。"""
        days = sorted({q.day for q in self._a})
        summary: dict[str, Any] = {}
        rows: list[dict[str, Any]] = []
        for d in days:
            active = [q for q in self._a if q.day == d and q.active]
            sts = [state["a"].get(q.key) for q in active]
            if not active or any(st is None or st["status"] != "ok" for st in sts):
                summary[d.isoformat()] = {"status": "incomplete"}
                continue
            sealed = [sealed_from_row(r) for st in sts if st is not None for r in st["rows"]]
            counts, per_ticker = coverage_compare(massive_on(sealed), tiingo_on(self._tiingo, d))
            summary[d.isoformat()] = {"status": "ok", **counts}
            rows.extend({"date": d.isoformat(), **r} for r in per_ticker)
        return summary, rows

    def _summary(self, state: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def strip(st: Mapping[str, Any] | None) -> dict[str, Any]:
            if st is None:
                return {"status": "not_run"}
            out = {k: v for k, v in st.items() if k not in ("rows", "profile")}
            return {**out, **st.get("profile", {})}

        coverage, rows = self._coverage(state)
        summary = {
            "a_type_queries": {q.key: strip(state["a"].get(q.key)) for q in self._a},
            "b_market_queries": {q.key: strip(state["b"].get(q.key)) for q in self._b},
            "b_stopped": bool(state.get("b_stopped")),
            "case_status": {
                c.key: state["cases"].get(c.key, {}).get("status", "not_run") for c in self._cases
            },
            "coverage_vs_tiingo": coverage,
            "requests_total": state["requests"],
            "elapsed_seconds_total": state["elapsed_seconds"],
        }
        return summary, rows
