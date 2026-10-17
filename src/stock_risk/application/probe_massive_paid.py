"""用例：Massive 付费档验证 `probe-massive-paid`（T08d-2b 第 1—3 节）。只验证，不做全量下载。

- 第 1 节：对象代码的日线（adjusted=false / true 各一次），统计、代码复用分段、
  与 Tiingo 对照；
- 第 2 节：拆股与分红（封存处理后保存），与 Tiingo 对照，自算拆股调整与 adjusted=true 比较；
- 第 3 节：5 个清单快照逐行保存，按规则 1—3 补主交易所与 type，抽样查 Ticker Overview 与日线首末日。
响应只在内存中做封存处理，原始响应不落盘；每完成一项保存进度，中断后续传。
日线请求被拒绝（HTTP 401 / 403）时视为付费档不支持历史价格：照实记录原文并停止（任务卡停止条件），
不换密钥重试；连续 3 次 429 保存进度后退出。照实记录，不作结论。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from stock_risk.application.ports import PriceSourceError, QueryRejected, RateLimitError
from stock_risk.application.progress import ProgressFn, quiet
from stock_risk.domain.massive_followup import TickerPages
from stock_risk.domain.massive_labels import (
    Snapshot,
    exchange_share,
    fill_field,
    label_report,
    paid_list_queries,
    pick_samples,
    seal_list_row,
    snapshot_key,
    unfilled_rows,
)
from stock_risk.domain.massive_paid import (
    COMPARE_TICKERS,
    PRICE_FROM,
    PRICE_TO,
    REUSE_TICKERS,
    Bar,
    Dividend,
    ReferenceAction,
    ReferenceBar,
    Split,
    bar_stats,
    compare_actions,
    compare_adjusted,
    compare_prices,
    paid_tickers,
    seal_bars,
    seal_dividend,
    seal_split,
    segments,
    split_adjusted_close,
)
from stock_risk.domain.massive_probe import ListQuery
from stock_risk.domain.seal import EXPLORATION_END

KINDS = ("raw", "adj")
PLAN_REJECT_STATUSES = (401, 403)
LIST_MAX_PAGES = 200


class PaidSource(Protocol):
    """Massive 付费档（只读）。原始记录只在内存中返回。"""

    @property
    def request_count(self) -> int: ...

    def aggs(self, ticker: str, start: date, end: date, adjusted: bool) -> TickerPages | None: ...

    def splits(self, ticker: str, until: date) -> TickerPages: ...

    def dividends(self, ticker: str, until: date) -> TickerPages: ...

    def list_tickers_pages(self, query: ListQuery, max_pages: int) -> TickerPages: ...

    def overview(self, ticker: str, day: date) -> dict[str, Any] | None: ...


class PaidStore(Protocol):
    """进度与输出。只接收封存处理后的行。"""

    def load_progress(self) -> dict[str, Any]: ...

    def save_progress(self, progress: Mapping[str, Any]) -> None: ...

    def save_bars(self, ticker: str, kind: str, bars: Sequence[Bar]) -> None: ...

    def load_bars(self, ticker: str, kind: str) -> list[Bar]: ...

    def save_actions(
        self, ticker: str, splits: Sequence[Split], dividends: Sequence[Dividend]
    ) -> None: ...

    def load_actions(self, ticker: str) -> tuple[list[Split], list[Dividend]]: ...

    def save_list(self, key: str, rows: Sequence[Mapping[str, Any]]) -> None: ...

    def load_list(self, key: str) -> list[dict[str, Any]]: ...

    def write_outputs(
        self,
        labels: Mapping[str, Sequence[Mapping[str, Any]]],
        samples: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> list[str]: ...


@dataclass(slots=True)
class PaidResult:
    summary: dict[str, Any]
    paths: list[str]
    stopped: str | None = None  # rate_limited / plan_rejected
    requests_now: int = 0
    remaining: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _Fail:
    """被拒绝或出错（照实记录的状态）。"""

    status: dict[str, Any]


class _Stop(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _status(kind: str, **extra: Any) -> dict[str, Any]:
    return {"status": kind, **extra}


class ProbeMassivePaidService:
    def __init__(
        self,
        source: PaidSource,
        store: PaidStore,
        calendar: Sequence[date],
        tiingo_prices: Mapping[str, Sequence[ReferenceBar]],
        tiingo_actions: Mapping[str, Sequence[ReferenceAction]],
        clock: Callable[[], float],
        progress: ProgressFn = quiet,
        tickers: Sequence[str] | None = None,
        list_queries: Sequence[ListQuery] | None = None,
    ) -> None:
        self._source = source
        self._store = store
        self._calendar = tuple(sorted(d for d in calendar if PRICE_FROM <= d <= PRICE_TO))
        self._tiingo_prices = tiingo_prices
        self._tiingo_actions = tiingo_actions
        self._clock = clock
        self._progress = progress
        self._tickers = tuple(tickers) if tickers is not None else paid_tickers()
        self._lists = tuple(list_queries) if list_queries is not None else paid_list_queries()

    def run(self) -> PaidResult:
        state = self._store.load_progress()
        for k in ("prices", "actions", "lists", "samples"):
            state.setdefault(k, {})
        state.setdefault("requests", 0)
        state.setdefault("elapsed_seconds", 0.0)
        before, t0 = self._source.request_count, self._clock()
        stopped: str | None = None
        try:
            self._run_prices(state)
            self._run_actions(state)
            self._run_lists(state)
            self._run_samples(state)
        except _Stop as e:
            stopped = e.reason
            self._progress(f"    停止：{e.reason}，进度已保存")
        finally:
            state["requests"] += self._source.request_count - before
            state["elapsed_seconds"] = round(state["elapsed_seconds"] + self._clock() - t0, 1)
            self._store.save_progress(state)
        summary, labels = self._summary(state)
        summary["stopped"] = stopped or state.get("plan_rejected") and "plan_rejected"
        paths = self._store.write_outputs(labels, self._sample_rows(state), summary)
        return PaidResult(
            summary, paths, stopped, self._source.request_count - before, self._remaining(state)
        )

    # ---- 请求与错误处理 ----

    def _call(self, fn: Callable[[], Any], price: bool = False) -> Any:
        """发出请求；429 用尽抛 _Stop；日线被 401/403 拒绝视为付费档拒绝历史价格，记录后停止。
        其余拒绝或错误返回 _Fail，照实记录，继续下一项。"""
        try:
            return fn()
        except RateLimitError:
            raise _Stop("rate_limited") from None
        except QueryRejected as e:
            self._progress(f"    被拒绝：HTTP {e.http_status} {e.message}")
            st = _status("rejected", http_status=e.http_status, message=e.message)
            if price and e.http_status in PLAN_REJECT_STATUSES:
                raise _PlanRejected(st) from None
            return _Fail(st)
        except PriceSourceError as e:
            self._progress(f"    失败：{e}")
            return _Fail(_status("error", message=str(e)))

    def _save(self, state: dict[str, Any]) -> None:
        self._store.save_progress(state)

    # ---- 第 1 节 ----

    def _run_prices(self, state: dict[str, Any]) -> None:
        if state.get("plan_rejected"):
            raise _Stop("plan_rejected")
        for t in self._tickers:
            for kind in KINDS:
                key = f"{t}_{kind}"
                if key in state["prices"]:
                    continue
                self._progress(f"[日线] {key}")
                try:
                    got = self._call(
                        lambda t=t, kind=kind: self._source.aggs(
                            t, PRICE_FROM, PRICE_TO, kind == "adj"
                        ),
                        price=True,
                    )
                except _PlanRejected as e:
                    state["prices"][key] = e.status
                    state["plan_rejected"] = e.status
                    self._save(state)
                    raise _Stop("plan_rejected") from None
                if isinstance(got, _Fail):
                    state["prices"][key] = got.status
                elif got is None:
                    state["prices"][key] = _status("not_found")
                elif not got.complete:
                    state["prices"][key] = _status("error", message=f"分页超过 {got.pages} 页")
                else:
                    bars, dropped = seal_bars(got.rows)
                    self._store.save_bars(t, kind, bars)
                    state["prices"][key] = _status(
                        "ok", pages=got.pages, rows=len(bars), dropped_after_2019=dropped
                    )
                    self._progress(f"    {len(bars)} 行，丢弃 {dropped} 行")
                self._save(state)

    # ---- 第 2 节 ----

    def _run_actions(self, state: dict[str, Any]) -> None:
        for t in self._tickers:
            if t in state["actions"]:
                continue
            self._progress(f"[拆股、分红] {t}")
            sp = self._call(lambda t=t: self._source.splits(t, EXPLORATION_END))
            dv = self._call(lambda t=t: self._source.dividends(t, EXPLORATION_END))
            if isinstance(sp, _Fail) or isinstance(dv, _Fail):
                state["actions"][t] = _status(
                    "rejected_or_error",
                    splits=sp.status if isinstance(sp, _Fail) else None,
                    dividends=dv.status if isinstance(dv, _Fail) else None,
                )
            else:
                splits = [s for s in (seal_split(r) for r in sp.rows) if s is not None]
                divs = [d for d in (seal_dividend(r) for r in dv.rows) if d is not None]
                self._store.save_actions(t, splits, divs)
                state["actions"][t] = _status(
                    "ok",
                    splits=len(splits),
                    dividends=len(divs),
                    splits_dropped_after_2019=len(sp.rows) - len(splits),
                    dividends_dropped_after_2019=len(dv.rows) - len(divs),
                    dividend_dates_masked=sum(len(d.masked_after_2019) for d in divs),
                )
                self._progress(f"    拆股 {len(splits)} 条，分红 {len(divs)} 条")
            self._save(state)

    # ---- 第 3 节 ----

    def _run_lists(self, state: dict[str, Any]) -> None:
        for q in self._lists:
            key = snapshot_key(q)
            if key in state["lists"]:
                continue
            self._progress(f"[清单] {key}")
            got = self._call(lambda q=q: self._source.list_tickers_pages(q, LIST_MAX_PAGES))
            if isinstance(got, _Fail):
                state["lists"][key] = got.status
            elif not got.complete:
                state["lists"][key] = _status("error", message=f"分页超过 {got.pages} 页")
            else:
                rows = [r for r in (seal_list_row(x) for x in got.rows) if r is not None]
                self._store.save_list(key, rows)
                state["lists"][key] = _status(
                    "ok",
                    pages=got.pages,
                    rows=len(rows),
                    dropped_list_date_after_2019=len(got.rows) - len(rows),
                    delisted_after_2019=sum(1 for r in rows if r["delisted_after_2019"]),
                )
                self._progress(f"    {len(rows)} 条，{got.pages} 页")
            self._save(state)

    def _snapshots(self, state: Mapping[str, Any]) -> list[Snapshot]:
        out = []
        for q in self._lists:
            key = snapshot_key(q)
            if state["lists"].get(key, {}).get("status") == "ok":
                out.append(Snapshot(key, q.day, tuple(self._store.load_list(key))))
        return out

    def _lists_done(self, state: Mapping[str, Any]) -> bool:
        return all(snapshot_key(q) in state["lists"] for q in self._lists)

    def _run_samples(self, state: dict[str, Any]) -> None:
        if not self._lists_done(state):
            return
        if "chosen" not in state["samples"]:
            snaps = self._snapshots(state)
            ex, ty = fill_field(snaps, "primary_exchange"), fill_field(snaps, "type")
            state["samples"]["chosen"] = pick_samples(
                unfilled_rows(snaps, ex), unfilled_rows(snaps, ty)
            )
            state["samples"]["done"] = {}
            self._save(state)
        for s in state["samples"]["chosen"]:
            key = f"{s['snapshot']}_{s['ticker']}"
            if key in state["samples"]["done"]:
                continue
            self._progress(f"[抽样] {key}")
            day = date.fromisoformat(s["date"])
            ov = self._call(lambda s=s, day=day: self._source.overview(s["ticker"], day))
            px = self._call(lambda s=s: self._source.aggs(s["ticker"], PRICE_FROM, PRICE_TO, False))
            state["samples"]["done"][key] = {
                "overview": _overview_result(ov),
                "prices": _sample_prices(px),
            }
            self._save(state)

    # ---- 汇总 ----

    def _remaining(self, state: Mapping[str, Any]) -> list[str]:
        out = [
            f"{t}_{k}" for t in self._tickers for k in KINDS if f"{t}_{k}" not in state["prices"]
        ]
        out += [t for t in self._tickers if t not in state["actions"]]
        out += [snapshot_key(q) for q in self._lists if snapshot_key(q) not in state["lists"]]
        chosen = state["samples"].get("chosen")
        if chosen is None:
            out.append("samples")
        else:
            done = state["samples"].get("done", {})
            out += [k for k in (f"{s['snapshot']}_{s['ticker']}" for s in chosen) if k not in done]
        return out

    def _sample_rows(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        done = state["samples"].get("done", {})
        rows = []
        for s in state["samples"].get("chosen", []):
            r = done.get(f"{s['snapshot']}_{s['ticker']}", {})
            ov, px = r.get("overview", {}), r.get("prices", {})
            rows.append(
                {
                    **s,
                    "overview_status": ov.get("status", "not_run"),
                    "sic_code": ov.get("sic_code"),
                    "overview_type": ov.get("type"),
                    "overview_primary_exchange": ov.get("primary_exchange"),
                    "prices_status": px.get("status", "not_run"),
                    "first": px.get("first"),
                    "last": px.get("last"),
                    "rows": px.get("rows"),
                }
            )
        return rows

    def _price_part(self, state: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for t in self._tickers:
            part: dict[str, Any] = {}
            for kind in KINDS:
                st = state["prices"].get(f"{t}_{kind}")
                if st is None:
                    part[kind] = {"status": "not_run"}
                elif st["status"] != "ok":
                    part[kind] = st
                else:
                    bars = self._store.load_bars(t, kind)
                    part[kind] = {**st, **bar_stats(bars, self._calendar)}
            raw_ok = state["prices"].get(f"{t}_raw", {}).get("status") == "ok"
            if raw_ok and t in REUSE_TICKERS:
                part["segments"] = segments(self._store.load_bars(t, "raw"), self._calendar)
            if raw_ok and t in COMPARE_TICKERS and t in self._tiingo_prices:
                part["vs_tiingo"] = compare_prices(
                    self._store.load_bars(t, "raw"), self._tiingo_prices[t]
                )
            out[t] = part
        return out

    def _action_part(self, state: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for t in self._tickers:
            st = state["actions"].get(t)
            if st is None or st["status"] != "ok":
                out[t] = st or {"status": "not_run"}
                continue
            splits, divs = self._store.load_actions(t)
            part: dict[str, Any] = {
                **st,
                "splits_2008_2019": sum(1 for s in splits if s.day >= PRICE_FROM),
                "dividends_2008_2019": sum(1 for d in divs if d.day >= PRICE_FROM),
            }
            if t in COMPARE_TICKERS and t in self._tiingo_actions:
                part["vs_tiingo"] = compare_actions(splits, divs, self._tiingo_actions[t])
            prices = state["prices"]
            if all(prices.get(f"{t}_{k}", {}).get("status") == "ok" for k in KINDS):
                mine = split_adjusted_close(self._store.load_bars(t, "raw"), splits)
                part["self_split_adjust_vs_adjusted"] = compare_adjusted(
                    mine, self._store.load_bars(t, "adj")
                )
            out[t] = part
        return out

    def _label_part(
        self, state: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
        part: dict[str, Any] = {
            "lists": {
                snapshot_key(q): state["lists"].get(snapshot_key(q), {"status": "not_run"})
                for q in self._lists
            },
        }
        labels: dict[str, list[dict[str, Any]]] = {}
        if not self._lists_done(state):
            return part, labels
        snaps = self._snapshots(state)
        ex, ty = fill_field(snaps, "primary_exchange"), fill_field(snaps, "type")
        report = label_report(snaps, ex, ty)
        part["fill"] = report
        first = snapshot_key(self._lists[0])
        if first in report:
            part["first_snapshot_exchange_share_xnas"] = exchange_share(report, first)
        for s in snaps:
            labels[s.key] = [
                {
                    "ticker": r.get("ticker"),
                    "cik": r.get("cik"),
                    "composite_figi": r.get("composite_figi"),
                    "primary_exchange": r.get("primary_exchange"),
                    "exchange_filled": ex[(s.key, i)].value,
                    "exchange_rule": ex[(s.key, i)].rule,
                    "exchange_conflict": "|".join(ex[(s.key, i)].conflict)
                    if len(ex[(s.key, i)].conflict) > 1
                    else "",
                    "type": r.get("type"),
                    "type_filled": ty[(s.key, i)].value,
                    "type_rule": ty[(s.key, i)].rule,
                    "type_conflict": "|".join(ty[(s.key, i)].conflict)
                    if len(ty[(s.key, i)].conflict) > 1
                    else "",
                }
                for i, r in enumerate(s.rows)
            ]
        return part, labels

    def _seal_counts(self, state: Mapping[str, Any]) -> dict[str, Any]:
        prices = [v for v in state["prices"].values() if v.get("status") == "ok"]
        acts = [v for v in state["actions"].values() if v.get("status") == "ok"]
        lists = [v for v in state["lists"].values() if v.get("status") == "ok"]
        samples = state["samples"].get("done", {}).values()
        return {
            "price_rows_dropped_after_2019": sum(v["dropped_after_2019"] for v in prices),
            "splits_dropped_after_2019": sum(v["splits_dropped_after_2019"] for v in acts),
            "dividends_dropped_after_2019": sum(v["dividends_dropped_after_2019"] for v in acts),
            "dividend_dates_masked": sum(v["dividend_dates_masked"] for v in acts),
            "list_rows_dropped_list_date_after_2019": sum(
                v["dropped_list_date_after_2019"] for v in lists
            ),
            "list_rows_delisted_after_2019": sum(v["delisted_after_2019"] for v in lists),
            "sample_overview_dropped_by_seal": sum(
                1 for s in samples if s["overview"].get("status") == "dropped_by_seal"
            ),
            "sample_price_rows_dropped_after_2019": sum(
                s["prices"].get("dropped_after_2019", 0) for s in samples
            ),
        }

    def _summary(
        self, state: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
        labels_part, labels = self._label_part(state)
        summary = {
            "plan_rejected": state.get("plan_rejected"),
            "prices": self._price_part(state),
            "actions": self._action_part(state),
            "labels": labels_part,
            "seal": self._seal_counts(state),
            "requests_total": state["requests"],
            "elapsed_seconds_total": state["elapsed_seconds"],
        }
        return summary, labels


class _PlanRejected(Exception):
    def __init__(self, status: dict[str, Any]) -> None:
        super().__init__("plan rejected")
        self.status = status


def _overview_result(ov: Any) -> dict[str, Any]:
    """抽样的 Ticker Overview：只记录 sic_code、type、primary_exchange。

    list_date 晚于截止日的按封存丢弃。
    """
    if isinstance(ov, _Fail):
        return ov.status  # 被拒绝或出错
    if ov is None:
        return _status("not_found")
    listed = ov.get("list_date")
    if listed and date.fromisoformat(str(listed)[:10]) > EXPLORATION_END:
        return _status("dropped_by_seal")
    return _status(
        "found",
        sic_code=ov.get("sic_code"),
        type=ov.get("type"),
        primary_exchange=ov.get("primary_exchange"),
    )


def _sample_prices(px: Any) -> dict[str, Any]:
    """抽样的日线：只记录首日、末日与行数，不保存价格。"""
    if isinstance(px, _Fail):
        return px.status
    if px is None:
        return _status("not_found")
    bars, dropped = seal_bars(px.rows)
    if not bars:
        return _status("empty", rows=0, dropped_after_2019=dropped)
    return _status(
        "ok",
        first=bars[0].day.isoformat(),
        last=bars[-1].day.isoformat(),
        rows=len(bars),
        dropped_after_2019=dropped,
    )
