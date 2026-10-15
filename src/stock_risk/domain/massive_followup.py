"""Massive 试验的追加探查（T08d-2a 第 2.4 节，2026-10-15 补签）：查询计划、计数与对照。纯函数。

- A：去掉类型过滤，4 个日期 × {XNYS, XNAS} × active 真 / 假，按 type 计数（空值记为"(空)"）；
- B：去掉交易所与类型过滤（market=stocks、active=true），按 primary_exchange × type 交叉计数；
- C：补充案例（Ticker Overview 加 date）；
- D：与 T08d-1 的 Tiingo 探查（data/universe_probe/）按代码对照，不发请求。
封存处理沿用 domain/massive_probe.py（第 2.3 节）。照实记录，不作结论。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from stock_risk.domain.massive_probe import (
    ACTIVE_FLAGS,
    EXCHANGES,
    QUERY_DATES,
    Case,
    ListQuery,
    SealedTicker,
)
from stock_risk.domain.seal import assert_dates_not_sealed

EMPTY = "(空)"
B_MARKET = "stocks"
B_FIRST_DATE_MAX_PAGES = 40  # 第 2.4 节：B 的第一个日期翻页超过 40 页时停止 B
TIINGO_EXCHANGES = ("NYSE", "NASDAQ")
COVERAGE_EXAMPLES = 10


@dataclass(frozen=True, slots=True)
class TickerPages:
    """All Tickers 一次查询翻页的结果。complete 为假表示达到页数上限时仍有下一页。"""

    rows: list[dict[str, Any]]
    pages: int
    complete: bool


def followup_type_queries() -> tuple[ListQuery, ...]:
    """A：不传 type。4 个日期 × 2 个交易所 × active 真 / 假，共 16 种。"""
    return tuple(
        ListQuery(d, x, None, a) for d in QUERY_DATES for x in EXCHANGES for a in ACTIVE_FLAGS
    )


def followup_market_queries() -> tuple[ListQuery, ...]:
    """B：market=stocks、active=true，不传 exchange 与 type。4 个日期。"""
    return tuple(ListQuery(d, None, None, True, market=B_MARKET) for d in QUERY_DATES)


def _fc(ticker: str, day: str, note: str) -> Case:
    return Case("补充案例", ticker, date.fromisoformat(day), note)


FOLLOWUP_CASES: tuple[Case, ...] = (
    _fc("MSFT", "2010-01-04", "2010 年 Nasdaq 大盘股能否查到，主交易所与 type"),
    _fc("AAPL", "2010-01-04", "同上"),
    _fc("CME", "2010-01-04", "第一次运行显示 2010 年为 XNYS；产品经理印象中 2008 年已转 Nasdaq"),
    _fc("BIDU", "2012-06-01", "Nasdaq 上市的 ADR，type 是否为 ADRC"),
    _fc("SNY", "2015-06-01", "NYSE 上市的 ADR，type 是否为 ADRC"),
    _fc("BABA", "2015-06-01", "NYSE 上市的 ADR（2014 年上市）"),
)


def _label(v: Any) -> str:
    return EMPTY if v is None or v == "" else str(v)


def type_profile(rows: Sequence[SealedTicker], inactive: bool) -> dict[str, Any]:
    """A 的一种组合：按 type 计数（空值为"(空)"），cik、composite_figi 非空条数；
    active=false 的另报 delisted_utc 非空条数（封存后，含 delisted_after_2019 标记）。"""
    by_type: dict[str, int] = {}
    for r in rows:
        t = _label(r.get("type"))
        by_type[t] = by_type.get(t, 0) + 1
    out: dict[str, Any] = {
        "count": len(rows),
        "by_type": dict(sorted(by_type.items())),
        "cik_nonempty": sum(1 for r in rows if _label(r.get("cik")) != EMPTY),
        "composite_figi_nonempty": sum(1 for r in rows if _label(r.get("composite_figi")) != EMPTY),
    }
    if inactive:
        out["delisted_nonempty"] = sum(
            1 for r in rows if r.delisted is not None or r.delisted_after_2019
        )
        out["delisted_after_2019"] = sum(1 for r in rows if r.delisted_after_2019)
    return out


def crosstab(rows: Iterable[SealedTicker]) -> dict[str, dict[str, int]]:
    """B：primary_exchange × type 交叉计数，空值为"(空)"。返回 {交易所: {type: 条数}}。"""
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        row = out.setdefault(_label(r.get("primary_exchange")), {})
        t = _label(r.get("type"))
        row[t] = row.get(t, 0) + 1
    return {x: dict(sorted(ts.items())) for x, ts in sorted(out.items())}


def merge_crosstab(
    a: Mapping[str, Mapping[str, int]], b: Mapping[str, Mapping[str, int]]
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {x: dict(ts) for x, ts in a.items()}
    for x, ts in b.items():
        row = out.setdefault(x, {})
        for t, n in ts.items():
            row[t] = row.get(t, 0) + n
    return {x: dict(sorted(ts.items())) for x, ts in sorted(out.items())}


@dataclass(frozen=True, slots=True)
class ReferenceSpan:
    """T08d-1 Tiingo 探查输出的一行（已筛选 NYSE / NASDAQ、Stock、USD，已封存处理）。"""

    ticker: str
    exchange: str
    start: date
    end: date  # 2019 年后结束的已写为 2019-12-31（active_after_2019）
    active_after_2019: bool

    def __post_init__(self) -> None:
        assert_dates_not_sealed(f"Tiingo 代码清单 {self.ticker}", self.start, self.end)


def tiingo_on(spans: Iterable[ReferenceSpan], day: date) -> dict[str, set[str]]:
    """D：该日期 Tiingo 中 exchange ∈ {NYSE, NASDAQ}、开始日 ≤ 该日期 ≤ 结束日的代码 → 交易所。
    结束日只有"2019 年后结束"标记的（end = 2019-12-31）视为覆盖到该日期。"""
    out: dict[str, set[str]] = {}
    for s in spans:
        if s.exchange not in TIINGO_EXCHANGES:
            continue
        if s.start <= day and (day <= s.end or s.active_after_2019):
            out.setdefault(s.ticker, set()).add(s.exchange)
    return out


def massive_on(rows: Iterable[SealedTicker]) -> dict[str, dict[str, set[str]]]:
    """D：A 中 active=true 的行（两个交易所合并，不论 type）→ {代码: {exchange, type}}。"""
    out: dict[str, dict[str, set[str]]] = {}
    for r in rows:
        d = out.setdefault(str(r.get("ticker")), {"exchange": set(), "type": set()})
        d["exchange"].add(_label(r.get("primary_exchange")))
        d["type"].add(_label(r.get("type")))
    return out


def coverage_compare(
    massive: Mapping[str, Mapping[str, set[str]]],
    tiingo: Mapping[str, set[str]],
    examples: int = COVERAGE_EXAMPLES,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """D：按代码比较。返回（计数与各类前 examples 个例子，逐代码的对照行）。"""

    def row(t: str, side: str) -> dict[str, str]:
        m = massive.get(t, {})
        return {
            "ticker": t,
            "side": side,
            "massive_exchange": "|".join(sorted(m.get("exchange", ()))),
            "massive_type": "|".join(sorted(m.get("type", ()))),
            "tiingo_exchange": "|".join(sorted(tiingo.get(t, ()))),
        }

    m, ti = set(massive), set(tiingo)
    groups = {"both": m & ti, "massive_only": m - ti, "tiingo_only": ti - m}
    rows = [row(t, side) for side, ts in groups.items() for t in sorted(ts)]
    summary = {
        "counts": {side: len(ts) for side, ts in groups.items()},
        "examples": {
            side: [row(t, side) for t in sorted(ts)[:examples]] for side, ts in groups.items()
        },
    }
    return summary, rows
