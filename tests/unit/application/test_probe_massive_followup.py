"""T08d-2a 第 2.4 节：probe-massive --followup（假数据源，不访问网络）。

查询计划（不传 type、exchange）；按 type 计数与交叉计数，空值计为"(空)"；与 Tiingo 对照的集合计算；
B 第一个日期超过页数上限时停止 B；B 不保存逐行内容；封存处理与续传。
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from stock_risk.application.ports import RateLimitError
from stock_risk.application.probe_massive_followup import ProbeMassiveFollowupService
from stock_risk.domain.massive_followup import (
    EMPTY,
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
from stock_risk.domain.massive_probe import Case, ListQuery, list_queries, seal_all
from stock_risk.infrastructure.storage.massive_followup_files import (
    MassiveFollowupFiles,
    read_reference_spans,
)

D1, D2 = date(2012, 6, 1), date(2015, 6, 1)
SECRET_WORDS = ("Secret Corp", "2023-", "2021-", "desc-text", "2020-")


def rec(ticker: str, exch: str | None, typ: str | None = "CS", **kw: object) -> dict:
    out: dict = {
        "ticker": ticker,
        "name": "Secret Corp",
        "primary_exchange": exch,
        "active": True,
        "cik": "0000000001",
        "composite_figi": "BBG" + ticker,
        "last_updated_utc": "2023-01-01T00:00:00Z",
        "description": "desc-text",
        **kw,
    }
    if typ is not None:
        out["type"] = typ
    return out


def span(t: str, x: str, start: str, end: str, after: bool = False) -> ReferenceSpan:
    return ReferenceSpan(t, x, date.fromisoformat(start), date.fromisoformat(end), after)


# ---- 领域：查询计划与计数 ----


def test_followup_query_plan() -> None:
    a, b = followup_type_queries(), followup_market_queries()
    assert len(a) == 16 and all(q.type is None and q.exchange for q in a)
    assert len(b) == 4 and all(q.type is None and q.exchange is None for q in b)
    assert all(q.market == "stocks" and q.active for q in b)
    keys = {q.key for q in (*a, *b)}
    assert len(keys) == 20 and not keys & {q.key for q in list_queries()}
    assert len(FOLLOWUP_CASES) == 6 and all(c.day <= date(2019, 12, 31) for c in FOLLOWUP_CASES)


def test_type_profile_counts_empty_and_delisted() -> None:
    rows = seal_all(
        [
            rec("A", "XNAS", "CS"),
            rec("B", "XNAS", None, cik=None),
            rec("C", "XNAS", "", composite_figi="", delisted_utc="2014-05-01"),
            rec("D", "XNAS", "ADRC", delisted_utc="2022-01-03"),
        ]
    )
    p = type_profile(rows, inactive=True)
    assert p["count"] == 4
    assert p["by_type"] == {EMPTY: 2, "ADRC": 1, "CS": 1}
    assert p["cik_nonempty"] == 3 and p["composite_figi_nonempty"] == 3
    assert p["delisted_nonempty"] == 2 and p["delisted_after_2019"] == 1
    assert "delisted_nonempty" not in type_profile(rows, inactive=False)


def test_crosstab_with_empty_values() -> None:
    rows = seal_all(
        [
            rec("A", "XNAS", "CS"),
            rec("B", "XNAS", None),
            rec("C", None, "CS"),
            rec("D", "XNGS", None),
            rec("E", "XNAS", None),
        ]
    )
    assert crosstab(rows) == {
        EMPTY: {"CS": 1},
        "XNAS": {EMPTY: 2, "CS": 1},
        "XNGS": {EMPTY: 1},
    }


def test_coverage_sets() -> None:
    spans = [
        span("AAA", "NYSE", "2000-01-03", "2019-12-31", after=True),  # 2019 年后结束：覆盖
        span("BBB", "NASDAQ", "2013-01-02", "2016-05-02"),  # 开始日晚于 D1
        span("CCC", "NASDAQ", "2008-01-02", "2012-06-01"),  # 结束日 = 当日：覆盖
        span("DDD", "NYSE MKT", "2000-01-03", "2019-12-31"),  # 交易所不在范围
        span("EEE", "NYSE", "2005-01-03", "2011-03-01"),  # 已结束
    ]
    ti = tiingo_on(spans, D1)
    assert ti == {"AAA": {"NYSE"}, "CCC": {"NASDAQ"}}
    m = massive_on(seal_all([rec("AAA", "XNYS"), rec("ZZZ", "XNAS", None)]))
    counts, rows = coverage_compare(m, ti, examples=1)
    assert counts["counts"] == {"both": 1, "massive_only": 1, "tiingo_only": 1}
    assert counts["examples"]["massive_only"] == [
        {
            "ticker": "ZZZ",
            "side": "massive_only",
            "massive_exchange": "XNAS",
            "massive_type": EMPTY,
            "tiingo_exchange": "",
        }
    ]
    assert [(r["ticker"], r["side"]) for r in rows] == [
        ("AAA", "both"),
        ("ZZZ", "massive_only"),
        ("CCC", "tiingo_only"),
    ]


def test_read_reference_spans(tmp_path: Path) -> None:
    p = tmp_path / "tickers_2008_2019.csv"
    p.write_text(
        "ticker,exchange,start_date,end_date,active_after_2019\n"
        "A,NYSE,1999-11-18,2019-12-31,true\nB,NASDAQ,2010-01-04,2014-03-03,false\n",
        "utf-8",
    )
    a, b = read_reference_spans(p)
    assert a.active_after_2019 and a.end == date(2019, 12, 31)
    assert not b.active_after_2019 and b.exchange == "NASDAQ"


# ---- 用例 ----

A_QS = [
    ListQuery(D1, "XNYS", None, True),
    ListQuery(D1, "XNAS", None, True),
    ListQuery(D1, "XNAS", None, False),
]
B_QS = [ListQuery(D1, None, None, True, market="stocks"), ListQuery(D2, None, None, True, "stocks")]
C_LIST = [Case("补充案例", "MSFT", date(2010, 1, 4), "x")]


class FakeSource:
    """lists[key] 为 TickerPages 或异常；记录调用与页数上限。"""

    def __init__(self, lists: dict, cases: dict) -> None:
        self.lists, self.cases = lists, cases
        self._n = 0
        self.calls: list[tuple[str, int]] = []

    @property
    def request_count(self) -> int:
        return self._n

    def list_tickers_pages(self, q: ListQuery, max_pages: int) -> TickerPages:
        self.calls.append((q.key, max_pages))
        v = self.lists[q.key]
        if isinstance(v, BaseException):
            self._n += 1
            raise v
        self._n += v.pages
        return v

    def overview(self, ticker: str, day: date):  # type: ignore[no-untyped-def]
        self._n += 1
        self.calls.append((f"{ticker}_{day}", 0))
        v = self.cases[f"{ticker}_{day.isoformat()}"]
        if isinstance(v, BaseException):
            raise v
        return v


def svc(out: Path, src: FakeSource, spans: list[ReferenceSpan]) -> ProbeMassiveFollowupService:
    return ProbeMassiveFollowupService(
        src,
        MassiveFollowupFiles(out),
        spans,
        clock=lambda: 0.0,
        type_queries=A_QS,
        market_queries=B_QS,
        cases=C_LIST,
        b_first_max_pages=2,
    )


def _lists(b_first: TickerPages) -> dict:
    return {
        A_QS[0].key: TickerPages(
            [rec("AAA", "XNYS"), rec("NEW", "XNYS", list_date="2021-02-01")], 1, True
        ),
        A_QS[1].key: TickerPages([rec("ZZZ", "XNAS", None)], 1, True),
        A_QS[2].key: TickerPages([rec("OLD", "XNAS", None, delisted_utc="2023-04-03")], 1, True),
        B_QS[0].key: b_first,
        B_QS[1].key: TickerPages([rec("Q", "XNGS", None)], 1, True),
    }


def test_followup_run_outputs_and_b_stop(tmp_path: Path) -> None:
    b_first = TickerPages([rec("P1", "XNAS", None), rec("P2", None, "CS")], 2, complete=False)
    src = FakeSource(_lists(b_first), {"MSFT_2010-01-04": rec("MSFT", "XNAS", None)})
    spans = [
        span("AAA", "NYSE", "2000-01-03", "2019-12-31", True),
        span("CCC", "NASDAQ", "2008-01-02", "2014-01-02"),
    ]
    r = svc(tmp_path, src, spans).run()
    s = r.summary
    assert not r.rate_limited and r.remaining == []
    # A：list_date 晚于截止日的行丢弃不计数；空 type 计为"(空)"
    a0 = s["a_type_queries"][A_QS[0].key]
    assert a0["status"] == "ok" and a0["count"] == 1 and a0["by_type"] == {"CS": 1}
    a2 = s["a_type_queries"][A_QS[2].key]
    assert a2["by_type"] == {EMPTY: 1} and a2["delisted_after_2019"] == 1
    # B：第一个日期超过上限 → 记录页数与已得计数，其余日期跳过
    b0 = s["b_market_queries"][B_QS[0].key]
    assert b0["status"] == "stopped_at_page_limit" and b0["pages"] == 2 and b0["count"] == 2
    assert b0["crosstab"] == {EMPTY: {"CS": 1}, "XNAS": {EMPTY: 1}}
    assert s["b_market_queries"][B_QS[1].key]["status"] == "skipped" and s["b_stopped"]
    assert (B_QS[0].key, 2) in src.calls and all(k != B_QS[1].key for k, _ in src.calls)
    # D：active=true 两个交易所合并
    cov = s["coverage_vs_tiingo"][D1.isoformat()]
    assert cov["counts"] == {"both": 1, "massive_only": 1, "tiingo_only": 1}
    assert s["case_status"] == {"MSFT_2010-01-04": "found"}
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [
        "_progress.json",
        "cases.csv",
        "coverage.csv",
        "crosstab_2012-06-01.csv",
        "lists_2012-06-01.csv",
        "summary.json",
    ]
    text = "".join(p.read_text("utf-8") for p in tmp_path.iterdir())
    for w in SECRET_WORDS:
        assert w not in text, w
    assert "P1" not in text and "P2" not in text  # B 不保存逐行内容
    ct = (tmp_path / "crosstab_2012-06-01.csv").read_text("utf-8").splitlines()
    assert ct == ["primary_exchange,type,count", f"{EMPTY},CS,1", f"XNAS,{EMPTY},1"]


def test_followup_b_continues_when_first_date_within_cap_and_resumes(tmp_path: Path) -> None:
    lists = _lists(TickerPages([rec("P1", "XNAS", None)], 2, complete=True))
    lists[A_QS[1].key] = RateLimitError("429")
    cases = {"MSFT_2010-01-04": None}
    r = svc(tmp_path, FakeSource(lists, cases), []).run()
    assert r.rate_limited and A_QS[1].key in r.remaining
    assert r.summary["coverage_vs_tiingo"][D1.isoformat()] == {"status": "incomplete"}
    progress = json.loads((tmp_path / "_progress.json").read_text("utf-8"))
    assert A_QS[0].key in progress["a"] and A_QS[1].key not in progress["a"]
    lists[A_QS[1].key] = TickerPages([rec("ZZZ", "XNAS", None)], 1, True)
    src2 = FakeSource(lists, cases)
    r2 = svc(tmp_path, src2, []).run()
    assert src2.calls[0][0] == A_QS[1].key and A_QS[0].key not in [k for k, _ in src2.calls]
    assert not r2.rate_limited and r2.remaining == []
    b = r2.summary["b_market_queries"]
    assert b[B_QS[0].key]["status"] == "ok" and b[B_QS[1].key]["status"] == "ok"
    assert b[B_QS[1].key]["crosstab"] == {"XNGS": {EMPTY: 1}}
    assert r2.summary["case_status"] == {"MSFT_2010-01-04": "not_found"}


def test_request_count_saved_with_each_progress(tmp_path: Path) -> None:
    """每次保存进度都累计请求数：进程在中途被终止，已发生的请求也已记入进度。"""
    seen: list[int] = []
    store = MassiveFollowupFiles(tmp_path)
    save = store.save_progress

    def spy(progress: dict) -> None:
        seen.append(progress["requests"])
        save(progress)

    store.save_progress = spy  # type: ignore[method-assign]
    lists = _lists(TickerPages([rec("P1", "XNAS", None)], 2, complete=True))
    src = FakeSource(lists, {"MSFT_2010-01-04": None})
    service = svc(tmp_path, src, [])
    service._store = store
    r = service.run()
    assert seen[0] == 1 and seen == sorted(seen) and seen[-1] == src.request_count
    assert r.summary["requests_total"] == src.request_count
