"""T08d-2a：Massive 适配器（假传输层，不访问网络，不用真实密钥）。

限频与 429 退避；分页；date 晚于 2019-12-31 时拒绝执行；拒绝时记录状态与原文；
URL 与异常信息中不出现密钥。
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
from datetime import date

import pytest

from stock_risk.application.ports import PriceSourceError, QueryRejected, RateLimitError
from stock_risk.domain.massive_probe import ListQuery
from stock_risk.domain.seal import SealViolation
from stock_risk.infrastructure.massive.client import MassiveReference

KEY = "fake-massive-key-123"
Q = ListQuery(date(2015, 6, 1), "XNYS", "CS", False)


class Transport:
    """按顺序给出响应：dict（JSON）或 (HTTP 状态, 正文) 元组。记录收到的请求。"""

    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.requests: list[urllib.request.Request] = []

    def __call__(self, req):  # type: ignore[no-untyped-def]
        self.requests.append(req)
        r = self.responses.pop(0)
        if isinstance(r, tuple):
            code, body = r
            raise urllib.error.HTTPError(req.full_url, code, "x", {}, io.BytesIO(body.encode()))
        return r


class Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def client(transport: Transport, clock: Clock | None = None, **kw) -> MassiveReference:  # type: ignore[no-untyped-def]
    clock = clock or Clock()
    return MassiveReference(KEY, transport=transport, sleep=clock.sleep, clock=clock.now, **kw)


def test_auth_in_header_and_query_params_without_key() -> None:
    t = Transport({"status": "OK", "results": [{"ticker": "A"}]})
    assert client(t).list_tickers(Q) == [{"ticker": "A"}]
    (req,) = t.requests
    assert req.get_header("Authorization") == f"Bearer {KEY}"
    assert KEY not in req.full_url
    qs = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(req.full_url).query))
    assert qs == {
        "exchange": "XNYS",
        "type": "CS",
        "active": "false",
        "date": "2015-06-01",
        "limit": "1000",
        "order": "asc",
        "sort": "ticker",
    }


def test_pagination_follows_next_url_and_strips_api_key() -> None:
    nxt = f"https://api.massive.com/v3/reference/tickers?cursor=abc&apiKey={KEY}"
    t = Transport(
        {"status": "OK", "results": [{"ticker": "A"}], "next_url": nxt},
        {"status": "OK", "results": [{"ticker": "B"}]},
    )
    rows = client(t).list_tickers(Q)
    assert [r["ticker"] for r in rows] == ["A", "B"]
    second = t.requests[1].full_url
    assert "cursor=abc" in second and KEY not in second and "apiKey" not in second


def test_next_url_other_host_refused() -> None:
    t = Transport({"status": "OK", "results": [], "next_url": "https://evil.example/x"})
    with pytest.raises(PriceSourceError, match="api.massive.com"):
        client(t).list_tickers(Q)


def test_throttle_default_five_per_minute() -> None:
    clock = Clock()
    t = Transport(*[{"status": "OK", "results": []}] * 3)
    c = client(t, clock)
    for _ in range(3):
        c.list_tickers(Q)
    assert clock.sleeps == [12.0, 12.0] and c.request_count == 3


def test_429_backoff_then_success_and_three_429_raise() -> None:
    clock = Clock()
    t = Transport((429, ""), {"status": "OK", "results": [{"ticker": "A"}]})
    c = client(t, clock, per_minute=60)
    assert c.list_tickers(Q) == [{"ticker": "A"}]
    assert 60.0 in clock.sleeps and c.request_count == 2
    t2 = Transport((429, ""), (429, ""), (429, ""))
    c2 = client(t2, Clock())
    with pytest.raises(RateLimitError) as e:
        c2.list_tickers(Q)
    assert c2.request_count == 3 and KEY not in str(e.value)


def test_rejection_records_status_and_message_without_key() -> None:
    body = json.dumps(
        {"status": "NOT_AUTHORIZED", "message": f"Your plan doesn't include this data ({KEY})"}
    )
    t = Transport((403, body))
    with pytest.raises(QueryRejected) as e:
        client(t).list_tickers(Q)
    assert e.value.http_status == 403
    assert "NOT_AUTHORIZED" in e.value.message and "plan" in e.value.message
    assert KEY not in str(e.value) and KEY not in e.value.message
    assert e.value.__cause__ is None and "http" not in str(e.value).lower().replace("http 403", "")


def test_status_error_in_200_body_is_rejection() -> None:
    t = Transport({"status": "ERROR", "message": "bad date"})
    with pytest.raises(QueryRejected, match="bad date"):
        client(t).list_tickers(Q)


def test_overview_not_found_returns_none_and_found_returns_results() -> None:
    t = Transport((404, "{}"), {"status": "OK", "results": {"ticker": "GM", "active": False}})
    c = client(t)
    assert c.overview("LEH", date(2008, 6, 2)) is None
    assert c.overview("GM", date(2009, 3, 2)) == {"ticker": "GM", "active": False}
    assert "date=2009-03-02" in t.requests[1].full_url


def test_date_after_seal_rejected_before_any_request() -> None:
    t = Transport()
    c = client(t)
    with pytest.raises(SealViolation):
        c.overview("MSFT", date(2020, 1, 2))
    with pytest.raises(SealViolation):
        ListQuery(date(2020, 1, 2), "XNYS", "CS", True)
    assert t.requests == [] and c.request_count == 0


def test_network_error_message_has_no_key() -> None:
    class Net:
        def __call__(self, req):  # type: ignore[no-untyped-def]
            raise urllib.error.URLError(f"refused {KEY}")

    c = MassiveReference(KEY, transport=Net(), sleep=lambda _: None, clock=lambda: 0.0)
    with pytest.raises(PriceSourceError) as e:
        c.list_tickers(Q)
    assert KEY not in str(e.value)


# ---- 第 2.4 节：type、exchange 可选；页数上限 ----


def _qs(req: urllib.request.Request) -> dict[str, str]:
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(req.full_url).query))


def test_optional_type_and_exchange_not_sent() -> None:
    t = Transport(*[{"status": "OK", "results": []}] * 2)
    c = client(t)
    c.list_tickers(ListQuery(date(2010, 1, 4), "XNAS", None, True))
    c.list_tickers(ListQuery(date(2010, 1, 4), None, None, True, market="stocks"))
    a, b = (_qs(r) for r in t.requests)
    assert "type" not in a and a["exchange"] == "XNAS" and "market" not in a
    assert "type" not in b and "exchange" not in b and b["market"] == "stocks"
    assert b["active"] == "true" and b["date"] == "2010-01-04"


def test_page_cap_reports_pages_and_incomplete() -> None:
    nxt = "https://api.massive.com/v3/reference/tickers?cursor=x"
    page = {"status": "OK", "results": [{"ticker": "A"}], "next_url": nxt}
    t = Transport(page, page, page)
    got = client(t).list_tickers_pages(Q, 2)
    assert got.pages == 2 and not got.complete and len(got.rows) == 2 and len(t.requests) == 2
    t2 = Transport({"status": "OK", "results": [{"ticker": "A"}]})
    done = client(t2).list_tickers_pages(Q, 1)
    assert done.pages == 1 and done.complete
