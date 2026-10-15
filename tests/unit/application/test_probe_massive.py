"""T08d-2a：probe-massive 用例（假数据源，不访问网络）。

封存处理（delisted_utc 改写、list_date 丢弃、字段丢弃、原始响应不落盘）；被拒绝照实记录并继续；
429 用尽保存进度后退出、再次运行续传；主交易所变化；日志中不出现密钥。
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from stock_risk.application.ports import QueryRejected, RateLimitError
from stock_risk.application.probe_massive import ProbeMassiveService
from stock_risk.domain.massive_probe import (
    CASES,
    Case,
    ListQuery,
    exchange_changes,
    list_queries,
    seal_ticker,
)
from stock_risk.domain.seal import SealViolation
from stock_risk.infrastructure.storage.massive_probe_files import MassiveProbeFiles

D1, D2 = date(2012, 6, 1), date(2015, 6, 1)
QUERIES = [ListQuery(D1, "XNYS", "CS", True), ListQuery(D2, "XNYS", "CS", True)]
CASE_LIST = [Case("对照", "MSFT", D1, "x"), Case("破产后转场外", "LEH", date(2008, 6, 2), "x")]
SECRET_WORDS = ("Microsoft", "2023-", "2021-", "desc-text", "2020-")


def rec(ticker: str, figi: str, exch: str, **kw: object) -> dict:
    return {
        "ticker": ticker,
        "name": "Microsoft Corp",
        "primary_exchange": exch,
        "type": "CS",
        "active": True,
        "composite_figi": figi,
        "share_class_figi": figi + "S",
        "cik": "0000789019",
        "last_updated_utc": "2023-01-01T00:00:00Z",
        "description": "desc-text",
        **kw,
    }


class FakeSource:
    def __init__(self, lists: dict, cases: dict) -> None:
        self.lists, self.cases = lists, cases
        self._n = 0
        self.calls: list[str] = []

    @property
    def request_count(self) -> int:
        return self._n

    def _hit(self, v):  # type: ignore[no-untyped-def]
        self._n += 1
        if isinstance(v, BaseException):
            raise v
        return v

    def list_tickers(self, q: ListQuery) -> list[dict]:
        self.calls.append(q.key)
        return self._hit(self.lists[q.key])

    def overview(self, ticker: str, day: date):  # type: ignore[no-untyped-def]
        self.calls.append(f"{ticker}_{day}")
        return self._hit(self.cases[f"{ticker}_{day.isoformat()}"])


def svc(out: Path, src: FakeSource, logs: list[str] | None = None) -> ProbeMassiveService:
    return ProbeMassiveService(
        src,
        MassiveProbeFiles(out),
        clock=lambda: 0.0,
        progress=(logs.append if logs is not None else lambda _: None),
        queries=QUERIES,
        cases=CASE_LIST,
    )


def test_query_plan_and_cases_within_seal() -> None:
    qs = list_queries()
    assert len(qs) == 32 and len({q.key for q in qs}) == 32
    assert len(CASES) == 21 and all(c.day <= date(2019, 12, 31) for c in CASES)
    with pytest.raises(SealViolation):
        Case("x", "X", date(2020, 1, 1), "")


def test_seal_ticker_rewrites_drops_and_whitelists() -> None:
    s = seal_ticker(rec("A", "F1", "XNYS", delisted_utc="2021-03-15T00:00:00Z"))
    assert s is not None and s.delisted is None and s.delisted_after_2019
    row = s.as_row()
    assert "name" not in row and "last_updated_utc" not in row and "description" not in row
    keep = seal_ticker(rec("B", "F2", "XNYS", delisted_utc="2016-03-01T00:00:00Z"))
    assert keep is not None and keep.delisted == date(2016, 3, 1) and not keep.delisted_after_2019
    assert seal_ticker(rec("C", "F3", "XNAS", list_date="2020-02-01")) is None


def test_exchange_changes_by_figi() -> None:
    a = seal_ticker(rec("A", "F1", "XNAS"))
    b = seal_ticker(rec("A", "F1", "XNYS"))
    c = seal_ticker(rec("C", "F3", "XNYS"))
    out = exchange_changes({D1: [a, c], D2: [b, c]})  # type: ignore[list-item]
    assert out["figi_seen_on_multiple_dates"] == 2 and out["figi_with_exchange_change"] == 1
    (ex,) = out["examples"]
    assert ex["primary_exchange_by_date"] == {D1.isoformat(): ["XNAS"], D2.isoformat(): ["XNYS"]}


def test_run_seals_outputs_records_rejections_and_writes_no_raw(tmp_path: Path) -> None:
    src = FakeSource(
        {
            QUERIES[0].key: [
                rec("MSFT", "F1", "XNAS", delisted_utc="2023-05-01"),
                rec("NEW", "F9", "XNYS", list_date="2021-01-04"),
            ],
            QUERIES[1].key: QueryRejected(403, "NOT_AUTHORIZED；plan does not include history"),
        },
        {
            "MSFT_2012-06-01": rec("MSFT", "F1", "XNAS", list_date="1986-03-13"),
            "LEH_2008-06-02": None,
        },
    )
    logs: list[str] = []
    r = svc(tmp_path, src, logs).run()
    assert not r.rate_limited and r.remaining == [] and r.requests_now == 4
    s = r.summary
    assert s["list_queries"][QUERIES[0].key] == {"status": "ok", "count": 1}
    assert s["list_queries"][QUERIES[1].key]["status"] == "rejected"
    assert s["list_queries"][QUERIES[1].key]["http_status"] == 403
    assert s["case_status"] == {"MSFT_2012-06-01": "found", "LEH_2008-06-02": "not_found"}
    assert s["delisted_after_2019_rows"] == 1
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["_progress.json", "cases.csv", "lists_2012-06-01.csv", "summary.json"]
    text = "".join(p.read_text("utf-8") for p in tmp_path.iterdir())
    for w in SECRET_WORDS:  # 名称、更新时间、描述与 2020 年以后的日期都不落盘
        assert w not in text, w
    lst = (tmp_path / "lists_2012-06-01.csv").read_text("utf-8").splitlines()
    assert lst[0].startswith("query_exchange,query_type,query_active,ticker")
    assert lst[1].endswith(",,true") and "NEW" not in "".join(lst)
    assert any("plan does not include history" in x for x in logs)


def test_rate_limit_saves_progress_then_resumes(tmp_path: Path) -> None:
    lists = {QUERIES[0].key: [rec("A", "F1", "XNYS")], QUERIES[1].key: RateLimitError("429")}
    cases = {"MSFT_2012-06-01": None, "LEH_2008-06-02": None}
    r = svc(tmp_path, FakeSource(lists, cases)).run()
    assert r.rate_limited and QUERIES[1].key in r.remaining
    progress = json.loads((tmp_path / "_progress.json").read_text("utf-8"))
    assert QUERIES[0].key in progress["lists"] and QUERIES[1].key not in progress["lists"]
    lists[QUERIES[1].key] = [rec("A", "F1", "XNYS")]
    src2 = FakeSource(lists, cases)
    r2 = svc(tmp_path, src2).run()
    assert src2.calls[0] == QUERIES[1].key and QUERIES[0].key not in src2.calls
    assert not r2.rate_limited and r2.remaining == []
    assert r2.summary["requests_total"] == 2 + 3
