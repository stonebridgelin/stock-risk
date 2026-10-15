"""用例：Massive 数据源试验 `probe-massive`（T08d-2a 第 2 节）。只做探查，不下载价格。

- 第 2.1 节：历史日期的代码清单（32 种查询组合，翻完全部分页）；
- 第 2.2 节：已知案例的 Ticker Overview。
响应只在内存中做封存处理（domain/massive_probe.py），原始响应不落盘。每完成一项即保存进度
（只含封存处理后的行与状态），中断后再次运行跳过已完成项；连续 3 次 429 保存进度后退出。
被数据源拒绝（例如免费档不支持历史日期）照实记录 HTTP 状态与原文，不算失败，继续下一项。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from stock_risk.application.ports import PriceSourceError, QueryRejected, RateLimitError
from stock_risk.application.progress import ProgressFn, quiet
from stock_risk.domain.massive_probe import (
    ADMITTED_FIELDS,
    CASES,
    Case,
    ListQuery,
    SealedTicker,
    exchange_changes,
    list_queries,
    seal_all,
    seal_ticker,
)


class ReferenceSource(Protocol):
    """Massive 参考数据（只读）。原始记录只在内存中返回。"""

    @property
    def request_count(self) -> int: ...

    def list_tickers(self, query: ListQuery) -> list[dict[str, Any]]:
        """All Tickers：翻完全部分页。拒绝时抛 QueryRejected，429 用尽抛 RateLimitError。"""
        ...

    def overview(self, ticker: str, day: date) -> dict[str, Any] | None:
        """Ticker Overview（带 date）；查不到（404）时返回 None。"""
        ...


class ProbeStore(Protocol):
    """进度与输出。只接收封存处理后的行。"""

    def load_progress(self) -> dict[str, Any]: ...

    def save_progress(self, progress: Mapping[str, Any]) -> None: ...

    def write_outputs(
        self,
        lists: Mapping[date, Sequence[Mapping[str, Any]]],
        cases: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> list[str]: ...


@dataclass(slots=True)
class MassiveProbeResult:
    summary: dict[str, Any]
    paths: list[str]
    rate_limited: bool = False
    requests_now: int = 0
    remaining: list[str] = field(default_factory=list)


def _status(kind: str, **extra: Any) -> dict[str, Any]:
    return {"status": kind, **extra}


class ProbeMassiveService:
    def __init__(
        self,
        source: ReferenceSource,
        store: ProbeStore,
        clock: Callable[[], float],
        progress: ProgressFn = quiet,
        queries: Sequence[ListQuery] | None = None,
        cases: Sequence[Case] = CASES,
    ) -> None:
        self._source = source
        self._store = store
        self._clock = clock
        self._progress = progress
        self._queries = tuple(queries) if queries is not None else list_queries()
        self._cases = tuple(cases)

    def run(self) -> MassiveProbeResult:
        state = self._store.load_progress()
        state.setdefault("lists", {})
        state.setdefault("cases", {})
        state.setdefault("requests", 0)
        state.setdefault("elapsed_seconds", 0.0)
        before, t0 = self._source.request_count, self._clock()
        rate_limited = False
        try:
            rate_limited = self._run_lists(state) or self._run_cases(state)
        finally:
            state["requests"] += self._source.request_count - before
            state["elapsed_seconds"] = round(state["elapsed_seconds"] + self._clock() - t0, 1)
            self._store.save_progress(state)
        summary = self._summary(state)
        paths = self._store.write_outputs(self._lists(state), self._case_rows(state), summary)
        remaining = [q.key for q in self._queries if q.key not in state["lists"]]
        remaining += [c.key for c in self._cases if c.key not in state["cases"]]
        return MassiveProbeResult(
            summary, paths, rate_limited, self._source.request_count - before, remaining
        )

    # ---- 第 2.1 节 ----

    def _run_lists(self, state: dict[str, Any]) -> bool:
        for q in self._queries:
            if q.key in state["lists"]:
                continue
            self._progress(f"[清单] {q.key}")
            try:
                rows = seal_all(self._source.list_tickers(q))
            except RateLimitError:
                self._progress("    连续 3 次 429：保存进度后退出，稍后再运行")
                return True
            except QueryRejected as e:
                state["lists"][q.key] = _status(
                    "rejected", http_status=e.http_status, message=e.message
                )
                self._progress(f"    被拒绝：HTTP {e.http_status} {e.message}")
            except PriceSourceError as e:
                state["lists"][q.key] = _status("error", message=str(e))
                self._progress(f"    失败：{e}")
            else:
                state["lists"][q.key] = _status(
                    "ok", count=len(rows), rows=[r.as_row() for r in rows]
                )
                self._progress(f"    {len(rows)} 条")
            self._store.save_progress(state)
        return False

    # ---- 第 2.2 节 ----

    def _run_cases(self, state: dict[str, Any]) -> bool:
        for c in self._cases:
            if c.key in state["cases"]:
                continue
            self._progress(f"[案例] {c.ticker} @ {c.day}")
            try:
                raw = self._source.overview(c.ticker, c.day)
            except RateLimitError:
                self._progress("    连续 3 次 429：保存进度后退出，稍后再运行")
                return True
            except QueryRejected as e:
                state["cases"][c.key] = _status(
                    "rejected", http_status=e.http_status, message=e.message
                )
            except PriceSourceError as e:
                state["cases"][c.key] = _status("error", message=str(e))
            else:
                state["cases"][c.key] = case_result(raw)
            self._progress(f"    {state['cases'][c.key]['status']}")
            self._store.save_progress(state)
        return False

    # ---- 输出 ----

    def _lists(self, state: Mapping[str, Any]) -> dict[date, list[dict[str, Any]]]:
        out: dict[date, list[dict[str, Any]]] = {}
        for q in self._queries:
            st = state["lists"].get(q.key)
            if st is None or st["status"] != "ok":
                continue
            query = {"query_exchange": q.exchange, "query_type": q.type, "query_active": q.active}
            out.setdefault(q.day, []).extend({**query, **r} for r in st["rows"])
        return out

    def _case_rows(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        return case_rows(self._cases, state["cases"])

    def _summary(self, state: Mapping[str, Any]) -> dict[str, Any]:
        combos = {}
        for q in self._queries:
            st = state["lists"].get(q.key)
            if st is None:
                combos[q.key] = {"status": "not_run"}
            else:
                combos[q.key] = {k: v for k, v in st.items() if k != "rows"}
        by_date: dict[date, list[SealedTicker]] = {}
        for d, rows in self._lists(state).items():
            by_date[d] = [sealed_from_row(r) for r in rows]
        return {
            "list_queries": combos,
            "case_status": {
                c.key: state["cases"].get(c.key, {}).get("status", "not_run") for c in self._cases
            },
            "exchange_changes": exchange_changes(by_date),
            "delisted_after_2019_rows": sum(
                1 for rs in by_date.values() for r in rs if r.delisted_after_2019
            ),
            "requests_total": state["requests"],
            "elapsed_seconds_total": state["elapsed_seconds"],
        }


def case_result(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Ticker Overview 的一个结果 → 进度中的状态（封存处理后）。"""
    if raw is None:
        return _status("not_found")
    sealed: SealedTicker | None = seal_ticker(raw)
    if sealed is None:
        return _status("dropped_by_seal")  # list_date 晚于 2019-12-31，按封存丢弃
    return _status("found", row=sealed.as_row())


def case_rows(cases: Sequence[Case], done: Mapping[str, Any]) -> list[dict[str, Any]]:
    """cases.csv 的行：每个案例一行，未运行的记 not_run。"""
    rows = []
    for c in cases:
        st = done.get(c.key)
        base = {"category": c.category, "ticker": c.ticker, "query_date": c.day.isoformat()}
        if st is None:
            rows.append({**base, "status": "not_run"})
            continue
        rows.append(
            {
                **base,
                "status": st["status"],
                "http_status": st.get("http_status"),
                "message": st.get("message"),
                **(st.get("row") or {}),
                "ticker": c.ticker,
            }
        )
    return rows


def sealed_from_row(r: Mapping[str, Any]) -> SealedTicker:
    """进度中保存的行（SealedTicker.as_row）还原为 SealedTicker。"""

    def day(v: Any) -> date | None:
        return date.fromisoformat(v) if v else None

    return SealedTicker(
        tuple((f, r[f]) for f in ADMITTED_FIELDS if r.get(f) is not None),
        day(r.get("list_date")),
        day(r.get("delisted")),
        bool(r.get("delisted_after_2019")),
    )
