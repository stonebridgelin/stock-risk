"""T08d-2c 判定：付费档的拆股与分红改走旧路径 /v3/reference/splits、/v3/reference/dividends
（假传输层，不访问网络，假密钥）。

- 请求路径与参数：ticker、截止日（不晚于 2019-12-31）、按日期升序、每页 1000；密钥只在请求头；
- 翻页：沿 next_url 取完全部页，地址中的 apiKey 去掉，主机不变；
- 按旧路径的字段名解析：拆股 execution_date、split_from、split_to；
  分红 cash_amount、ex_dividend_date、dividend_type、declaration_date、record_date、pay_date；
- T08d-2b 第 2 节的封存规则不变：执行日、除息日晚于截止日的行丢弃；
  分红的其他日期晚于截止日的写为空并记入 masked_after_2019；
- 写盘再读回，字段不变。
"""

from __future__ import annotations

import urllib.parse
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from stock_risk.domain.massive_paid import Dividend, Split, seal_dividend, seal_split
from stock_risk.domain.seal import SealViolation
from stock_risk.infrastructure.massive.client import MassiveReference
from stock_risk.infrastructure.storage.massive_paid_files import MassivePaidFiles

KEY = "fake-paid-key-v3"
RELAY = "https://relaynex.vip"
D = date


class Transport:
    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = list(responses)
        self.requests: list[Any] = []

    def __call__(self, req):  # type: ignore[no-untyped-def]
        self.requests.append(req)
        return self.responses.pop(0)


def client(t: Transport) -> MassiveReference:
    return MassiveReference(
        KEY, base_url=RELAY, transport=t, sleep=lambda _: None, clock=lambda: 0.0
    )


def _parts(req: Any) -> tuple[str | None, str, dict[str, str]]:
    p = urllib.parse.urlsplit(req.full_url)
    return p.hostname, p.path, dict(urllib.parse.parse_qsl(p.query))


# 旧路径的响应（字段名取自中转服务对 AAPL、KO 的实测返回，取值为公开记录）
SPLITS_PAGE_1 = {
    "status": "OK",
    "results": [
        {"execution_date": "1987-06-16", "split_from": 1, "split_to": 2, "ticker": "AAPL"},
        {"execution_date": "2000-06-21", "split_from": 1, "split_to": 2, "ticker": "AAPL"},
    ],
    "next_url": f"{RELAY}/v3/reference/splits?cursor=abc&apiKey={KEY}",
}
SPLITS_PAGE_2 = {
    "status": "OK",
    "results": [
        {"execution_date": "2014-06-09", "split_from": 1, "split_to": 7, "ticker": "AAPL"},
        {"execution_date": "2020-08-31", "split_from": 1, "split_to": 4, "ticker": "AAPL"},
    ],
}
DIVIDENDS_PAGE_1 = {
    "status": "OK",
    "results": [
        {
            "cash_amount": 0.39,
            "declaration_date": "2018-10-18",
            "dividend_type": "CD",
            "ex_dividend_date": "2018-11-29",
            "pay_date": "2018-12-14",
            "record_date": "2018-11-30",
            "ticker": "KO",
        },
    ],
    "next_url": f"{RELAY}/v3/reference/dividends?cursor=def",
}
DIVIDENDS_PAGE_2 = {
    "status": "OK",
    "results": [
        {
            "cash_amount": 0.4,
            "declaration_date": "2019-10-17",
            "dividend_type": "CD",
            "ex_dividend_date": "2019-11-29",
            "pay_date": "2020-01-15",  # 晚于截止日：写为空
            "record_date": "2019-12-02",
            "ticker": "KO",
        },
        {"cash_amount": 0.41, "dividend_type": "CD", "ex_dividend_date": "2020-03-13"},
    ],
}


def test_splits_old_path_paginates_and_parses() -> None:
    t = Transport(SPLITS_PAGE_1, SPLITS_PAGE_2)
    got = client(t).splits("AAPL", D(2019, 12, 31))
    assert got.complete and got.pages == 2 and len(got.rows) == 4
    (h1, p1, q1), (h2, p2, q2) = (_parts(r) for r in t.requests)
    assert (h1, p1) == ("relaynex.vip", "/v3/reference/splits")
    assert q1 == {
        "ticker": "AAPL",
        "execution_date.lte": "2019-12-31",
        "limit": "1000",
        "sort": "execution_date",
        "order": "asc",
    }
    assert (h2, p2, q2) == ("relaynex.vip", "/v3/reference/splits", {"cursor": "abc"})
    for r in t.requests:
        assert KEY not in r.full_url and r.get_header("Authorization") == f"Bearer {KEY}"
    splits = [s for s in (seal_split(r) for r in got.rows) if s is not None]
    assert splits == [
        Split(D(1987, 6, 16), 1.0, 2.0),
        Split(D(2000, 6, 21), 1.0, 2.0),
        Split(D(2014, 6, 9), 1.0, 7.0),
    ]  # 2020-08-31 的一行按封存丢弃
    assert splits[-1].factor == 7


def test_dividends_old_path_paginates_and_parses() -> None:
    t = Transport(DIVIDENDS_PAGE_1, DIVIDENDS_PAGE_2)
    got = client(t).dividends("KO", D(2019, 12, 31))
    assert got.complete and got.pages == 2 and len(got.rows) == 3
    (h1, p1, q1), (h2, p2, q2) = (_parts(r) for r in t.requests)
    assert (h1, p1) == ("relaynex.vip", "/v3/reference/dividends")
    assert q1 == {
        "ticker": "KO",
        "ex_dividend_date.lte": "2019-12-31",
        "limit": "1000",
        "sort": "ex_dividend_date",
        "order": "asc",
    }
    assert (h2, p2, q2) == ("relaynex.vip", "/v3/reference/dividends", {"cursor": "def"})
    divs = [d for d in (seal_dividend(r) for r in got.rows) if d is not None]
    first = Dividend(
        D(2018, 11, 29), 0.39, "CD", D(2018, 10, 18), D(2018, 11, 30), D(2018, 12, 14), ()
    )
    second = Dividend(
        D(2019, 11, 29), 0.4, "CD", D(2019, 10, 17), D(2019, 12, 2), None, ("pay_date",)
    )
    assert divs == [first, second]  # 2020-03-13 除息的一行按封存丢弃


def test_old_path_seal_rejects_late_query_without_request() -> None:
    t = Transport()
    c = client(t)
    with pytest.raises(SealViolation):
        c.splits("AAPL", D(2020, 1, 2))
    with pytest.raises(SealViolation):
        c.dividends("KO", D(2020, 1, 2))
    assert t.requests == []


def test_old_path_next_url_other_host_stops() -> None:
    page = {**SPLITS_PAGE_1, "next_url": "https://api.massive.com/v3/reference/splits?cursor=x"}
    t = Transport(page)
    with pytest.raises(Exception, match="relaynex.vip"):
        client(t).splits("AAPL", D(2019, 12, 31))
    assert len(t.requests) == 1


def test_actions_round_trip_through_files(tmp_path: Path) -> None:
    store = MassivePaidFiles(tmp_path, "relaynex.vip")
    splits = [Split(D(2014, 6, 9), 1.0, 7.0)]
    divs = [
        Dividend(D(2019, 11, 29), 0.4, "CD", D(2019, 10, 17), D(2019, 12, 2), None, ("pay_date",))
    ]
    store.save_actions("KO", splits, divs)
    assert store.load_actions("KO") == (splits, divs)
    header = (tmp_path / "actions" / "KO_dividends.csv").read_text("utf-8").splitlines()[0]
    assert header == (
        "ex_dividend_date,cash_amount,dividend_type,declaration_date,record_date,pay_date,"
        "masked_after_2019,source_host"
    )
