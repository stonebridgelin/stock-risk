"""T08d-2b：Massive 付费档验证（假传输层与假数据源，不访问网络，不用真实密钥）。

价格请求 to 晚于 2019-12-31 拒绝执行；晚于截止日的行丢弃并计数；分红其他日期字段的封存改写；
补标签规则 1—3 的顺序与矛盾检测；自算拆股调整；密钥不出现在日志与异常信息中；
付费档拒绝历史价格时停止。
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from stock_risk.application.ports import QueryRejected
from stock_risk.application.probe_massive_paid import ProbeMassivePaidService
from stock_risk.domain.massive_followup import TickerPages
from stock_risk.domain.massive_labels import (
    Snapshot,
    fill_field,
    label_report,
    paid_list_queries,
    pick_samples,
    seal_list_row,
    suffix_type,
    unfilled_rows,
)
from stock_risk.domain.massive_paid import (
    Bar,
    ReferenceAction,
    ReferenceBar,
    Split,
    bar_stats,
    check_price_range,
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
from stock_risk.domain.seal import SealViolation
from stock_risk.infrastructure.massive.client import MassiveReference
from stock_risk.infrastructure.storage.massive_paid_files import MassivePaidFiles

KEY = "fake-paid-key-xyz"
D = date


def t_ms(d: date) -> int:
    """美东零点（UTC 05:00）的毫秒时间戳。"""
    return int(datetime(d.year, d.month, d.day, 5, tzinfo=UTC).timestamp() * 1000)


def bar(d: date, c: float, v: float = 100.0) -> dict:
    return {"t": t_ms(d), "o": c, "h": c, "l": c, "c": c, "v": v, "vw": c, "n": 3, "otc": False}


# ---- 第 1 节：日线 ----


def test_paid_tickers_unique_and_include_compare() -> None:
    ts = paid_tickers()
    assert len(ts) == len(set(ts)) == 25
    assert {"MSFT", "XOM", "KO", "GM", "DELL", "ADT", "BABA"} <= set(ts)


def test_price_range_after_cutoff_refused() -> None:
    check_price_range(D(2008, 1, 2), D(2019, 12, 31))
    with pytest.raises(SealViolation):
        check_price_range(D(2008, 1, 2), D(2020, 1, 2))


def test_seal_bars_drops_after_cutoff_and_converts_timestamp() -> None:
    bars, dropped = seal_bars(
        [bar(D(2019, 12, 31), 2), bar(D(2020, 1, 2), 3), bar(D(2019, 12, 30), 1)]
    )
    assert dropped == 1 and [b.day for b in bars] == [D(2019, 12, 30), D(2019, 12, 31)]
    assert bars[0].trades == 3 and bars[0].vwap == 1


def _bars(days: list[date], vol: float = 1.0) -> list[Bar]:
    return [Bar(d, 1, 1, 1, 1.0, vol, 1, 1) for d in days]


CAL = [D(2010, 1, d) for d in (4, 5, 6, 7, 8, 11, 12, 13, 14, 15)]


def test_bar_stats_gaps_and_zero_volume() -> None:
    bars = _bars([CAL[0], CAL[1], CAL[5], CAL[9]])
    bars[1] = Bar(CAL[1], 1, 1, 1, 1, 0.0, 1, 1)
    s = bar_stats(bars, CAL)
    assert s["rows"] == 4 and s["first"] == "2010-01-04" and s["last"] == "2010-01-15"
    assert s["missing_days"] == 6 and s["zero_volume_days"] == 1
    assert s["longest_gap"] == {"from": "2010-01-06", "to": "2010-01-08", "days": 3}


def test_segments_split_at_long_gap() -> None:
    bars = _bars([CAL[0], CAL[1], CAL[6], CAL[7]])
    segs = segments(bars, CAL, min_gap=3)
    assert [(s["from"], s["to"], s["gap_to_next_days"]) for s in segs] == [
        ("2010-01-04", "2010-01-05", 4),
        ("2010-01-12", "2010-01-13", None),
    ]
    assert len(segments(bars, CAL, min_gap=5)) == 1


def test_compare_prices_with_tiingo() -> None:
    mine = [Bar(CAL[0], 1, 1, 1, 10.0, 100, 1, 1), Bar(CAL[1], 1, 1, 1, 10.5, 110, 1, 1)]
    mine.append(Bar(CAL[3], 1, 1, 1, 9.0, 90, 1, 1))
    ref = [ReferenceBar(CAL[0], 10.0, 100), ReferenceBar(CAL[1], 10.48, 100)]
    ref.append(ReferenceBar(CAL[2], 9.9, 100))
    c = compare_prices(mine, ref)
    assert c["common_days"] == 2 and c["only_massive"] == 1 and c["only_tiingo"] == 1
    assert c["close_abs_diff_max"] == pytest.approx(0.02) and c["close_diff_over_0_01_days"] == 1
    assert c["volume_rel_diff_median"] == pytest.approx(0.05)


# ---- 第 2 节：拆股与分红 ----


def test_seal_split_and_dividend() -> None:
    assert seal_split({"execution_date": "2020-08-31", "split_from": 1, "split_to": 4}) is None
    s = seal_split(
        {
            "execution_date": "2014-06-09",
            "split_from": 1,
            "split_to": 7,
            "adjustment_type": "forward_split",
            "historical_adjustment_factor": 0.01,
        }
    )
    assert s is not None and s.factor == 7
    assert seal_dividend({"ex_dividend_date": "2020-02-19", "cash_amount": 0.51}) is None
    d = seal_dividend(
        {
            "ex_dividend_date": "2019-11-20",
            "cash_amount": 0.51,
            "declaration_date": "2019-09-18",
            "record_date": "2019-11-21",
            "pay_date": "2020-01-09",
            "split_adjusted_cash_amount": 0.51,
            "historical_adjustment_factor": 0.9,
        }
    )
    assert d is not None and d.pay_date is None and d.masked_after_2019 == ("pay_date",)
    assert d.record_date == D(2019, 11, 21)


def test_split_adjustment_and_compare() -> None:
    raw = [Bar(D(2014, 6, 6), 1, 1, 1, 700.0, 1, 1, 1), Bar(D(2014, 6, 9), 1, 1, 1, 93.0, 1, 1, 1)]
    mine = split_adjusted_close(raw, [Split(D(2014, 6, 9), 1, 7, "forward_split")])
    assert mine == {D(2014, 6, 6): 100.0, D(2014, 6, 9): 93.0}
    adj = [Bar(D(2014, 6, 6), 1, 1, 1, 100.0, 1, 1, 1), Bar(D(2014, 6, 9), 1, 1, 1, 93.1, 1, 1, 1)]
    c = compare_adjusted(mine, adj)
    assert (
        c["common_days"] == 2 and c["days_over_1e-6"] == 1 and c["rel_diff_max_day"] == "2014-06-09"
    )


def test_compare_actions_with_tiingo() -> None:
    divs = [
        seal_dividend({"ex_dividend_date": "2008-02-19", "cash_amount": 0.11}),
        seal_dividend({"ex_dividend_date": "2008-05-13", "cash_amount": 0.12}),
        seal_dividend({"ex_dividend_date": "2008-08-19", "cash_amount": 0.11}),
        seal_dividend({"ex_dividend_date": "2003-01-02", "cash_amount": 0.08}),  # Tiingo 区间外
    ]
    ref = [
        ReferenceAction(D(2008, 2, 19), "dividend", 0.11),
        ReferenceAction(D(2008, 5, 13), "dividend", 0.11),
        ReferenceAction(D(2008, 11, 18), "dividend", 0.13),
        ReferenceAction(D(2014, 6, 9), "split", 7.0),
    ]
    c = compare_actions([Split(D(2014, 6, 9), 1, 7, None)], [d for d in divs if d], ref)
    dv = c["dividends"]
    assert (dv["match"], dv["same_date_value_differs"], dv["only_massive"], dv["only_tiingo"]) == (
        1,
        1,
        1,
        1,
    )
    assert c["splits"]["match"] == 1


# ---- 第 3 节：补标签 ----


def test_list_queries_and_seal_row() -> None:
    qs = paid_list_queries()
    assert len(qs) == 5 and all(q.exchange is None and q.type is None for q in qs)
    assert [q.active for q in qs] == [True, True, True, True, False]
    assert seal_list_row({"ticker": "N", "list_date": "2021-01-04"}) is None
    r = seal_list_row({"ticker": "A", "name": "X Corp", "delisted_utc": "2022-01-03T00:00:00Z"})
    assert (
        r is not None and "name" not in r and r["delisted_utc"] is None and r["delisted_after_2019"]
    )


def test_suffix_types() -> None:
    assert suffix_type("ABRpA") == "PFD" and suffix_type("BACpL") == "PFD"
    assert suffix_type("XYZ.WS") == "WARRANT" and suffix_type("XYZ.W") == "WARRANT"
    assert suffix_type("XYZ.U") == "UNIT" and suffix_type("XYZ.R") == "RIGHT"
    assert suffix_type("XYZr") == "RIGHT" and suffix_type("MSFT") is None


def _row(t: str, ex: str | None, ty: str | None, figi: str | None, cik: str | None) -> dict:
    return {"ticker": t, "primary_exchange": ex, "type": ty, "composite_figi": figi, "cik": cik}


def test_fill_rules_order_and_conflicts() -> None:
    s10 = Snapshot(
        "2010-01-04_active",
        D(2010, 1, 4),
        (
            _row("MSFT", None, None, "F1", "1"),  # 规则 1
            _row("OLD", None, None, None, "2"),  # 规则 2（无 FIGI）
            _row("ABCpA", None, None, "F9", None),  # FIGI 在别处无值 → 规则 3（type），交易所补不上
            _row("ZZZ", None, None, None, None),  # 都补不上
            _row("CME", None, None, "F3", "3"),  # 规则 1，其他快照取值矛盾
            _row("NOF", None, None, "F4", "4"),  # 有 FIGI 但别处无值：不走规则 2
        ),
    )
    s12 = Snapshot(
        "2012-06-01_active",
        D(2012, 6, 1),
        (
            _row("MSFT", "XNAS", "CS", "F1", "1"),
            _row("OLD", "XNAS", "CS", "F2", "2"),
            _row("CME", "XNYS", "CS", "F3", "3"),
            _row("NOF", "XNAS", "CS", None, "4"),
        ),
    )
    s19 = Snapshot("2019-06-03_active", D(2019, 6, 3), (_row("CME", "XNAS", "CS", "F3", "3"),))
    snaps = [s10, s12, s19]
    ex, ty = fill_field(snaps, "primary_exchange"), fill_field(snaps, "type")
    k = "2010-01-04_active"
    assert (ex[(k, 0)].rule, ex[(k, 0)].value) == ("rule1", "XNAS")
    assert (ex[(k, 1)].rule, ex[(k, 1)].value) == ("rule2", "XNAS")
    assert ex[(k, 2)].rule == "none" and (ty[(k, 2)].rule, ty[(k, 2)].value) == ("rule3", "PFD")
    assert ex[(k, 3)].rule == "none" and ty[(k, 3)].rule == "none"
    cme = ex[(k, 4)]
    assert (
        cme.rule == "rule1" and cme.value == "XNYS" and cme.conflict == ("XNAS", "XNYS")
    )  # 取最近
    assert ex[(k, 5)].rule == "none"
    rep = label_report(snaps, ex, ty)[k]
    assert rep["primary_exchange"]["filled_by"] == {"rule1": 2, "rule2": 1}
    assert rep["primary_exchange"]["unfilled"] == 3 and rep["primary_exchange"]["conflicts"] == 1
    assert rep["type"]["filled_by"] == {"rule1": 2, "rule2": 1, "rule3": 1}
    assert rep["primary_exchange"]["filled_values"] == {"XNAS": 2, "XNYS": 1}
    samples = pick_samples(unfilled_rows(snaps, ex), unfilled_rows(snaps, ty), n=2)
    assert [(s["group"], s["ticker"]) for s in samples] == [
        ("primary_exchange", "ABCpA"),
        ("primary_exchange", "NOF"),
        ("type", "ZZZ"),
    ]


# ---- 适配器：付费档端点 ----


class Transport:
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


def client(t: Transport) -> MassiveReference:
    return MassiveReference(KEY, transport=t, sleep=lambda _: None, clock=lambda: 0.0)


def _qs(req: urllib.request.Request) -> tuple[str, dict[str, str]]:
    parts = urllib.parse.urlsplit(req.full_url)
    return parts.path, dict(urllib.parse.parse_qsl(parts.query))


def test_client_aggs_splits_dividends_requests() -> None:
    t = Transport(
        {"status": "OK", "results": [bar(D(2008, 1, 2), 1)]},
        {"status": "OK", "results": []},
        {"status": "OK", "results": []},
    )
    c = client(t)
    got = c.aggs("MSFT", D(2008, 1, 2), D(2019, 12, 31), adjusted=False)
    assert got is not None and got.complete and len(got.rows) == 1
    c.splits("MSFT", D(2019, 12, 31))
    c.dividends("MSFT", D(2019, 12, 31))
    (p1, q1), (p2, q2), (p3, q3) = (_qs(r) for r in t.requests)
    assert (
        p1 == "/v2/aggs/ticker/MSFT/range/1/day/2008-01-02/2019-12-31" and q1["adjusted"] == "false"
    )
    assert p2 == "/stocks/v1/splits" and q2["execution_date.lte"] == "2019-12-31"
    assert p3 == "/stocks/v1/dividends" and q3["ex_dividend_date.lte"] == "2019-12-31"
    for r in t.requests:
        assert KEY not in r.full_url and r.get_header("Authorization") == f"Bearer {KEY}"
    with pytest.raises(SealViolation):
        c.aggs("MSFT", D(2008, 1, 2), D(2020, 1, 2), adjusted=False)
    with pytest.raises(SealViolation):
        c.dividends("MSFT", D(2020, 1, 2))
    assert len(t.requests) == 3  # 拒绝执行时不发请求


def test_client_rejection_message_hides_key() -> None:
    body = json.dumps(
        {"status": "NOT_AUTHORIZED", "message": f"plan does not include this; key {KEY}"}
    )
    t = Transport((403, body))
    with pytest.raises(QueryRejected) as e:
        client(t).aggs("MSFT", D(2008, 1, 2), D(2019, 12, 31), adjusted=False)
    assert (
        KEY not in str(e.value)
        and KEY not in e.value.message
        and "plan does not" in e.value.message
    )


# ---- 用例 ----


class FakeSource:
    def __init__(self, reject_prices: bool = False) -> None:
        self.reject_prices = reject_prices
        self._n = 0
        self.calls: list[str] = []

    @property
    def request_count(self) -> int:
        return self._n

    def _hit(self, name: str) -> None:
        self._n += 1
        self.calls.append(name)

    def aggs(self, ticker, start, end, adjusted):  # type: ignore[no-untyped-def]
        self._hit(f"aggs:{ticker}:{adjusted}")
        if self.reject_prices:
            raise QueryRejected(403, "NOT_AUTHORIZED；plan does not include history")
        f = 0.5 if adjusted else 1.0
        rows = [bar(D(2014, 6, 6), 700 * f / 7 if adjusted else 700), bar(D(2014, 6, 9), 93)]
        rows.append(bar(D(2020, 1, 2), 90))  # 晚于截止日：丢弃并计数
        return TickerPages(rows, 1, True)

    def splits(self, ticker, until):  # type: ignore[no-untyped-def]
        self._hit(f"splits:{ticker}")
        return TickerPages(
            [{"execution_date": "2014-06-09", "split_from": 1, "split_to": 7}], 1, True
        )

    def dividends(self, ticker, until):  # type: ignore[no-untyped-def]
        self._hit(f"dividends:{ticker}")
        rows = [{"ex_dividend_date": "2019-11-20", "cash_amount": 0.5, "pay_date": "2020-01-09"}]
        return TickerPages(rows, 1, True)

    def list_tickers_pages(self, q, max_pages):  # type: ignore[no-untyped-def]
        self._hit(f"list:{q.key}")
        rows = [
            {
                "ticker": "AAA",
                "primary_exchange": None,
                "type": None,
                "composite_figi": "F1",
                "name": "Secret Corp",
            },
            {"ticker": "BBB", "primary_exchange": None, "type": None, "composite_figi": None},
            {"ticker": "NEW", "list_date": "2021-03-01"},
        ]
        if q.day.year == 2019 and q.active:
            rows.append(
                {"ticker": "AAA", "primary_exchange": "XNAS", "type": "CS", "composite_figi": "F1"}
            )
        return TickerPages(rows, 1, True)

    def overview(self, ticker, day):  # type: ignore[no-untyped-def]
        self._hit(f"overview:{ticker}")
        return {"ticker": ticker, "sic_code": "6770", "type": None, "name": "Secret Corp"}


LISTS = [
    ListQuery(D(2010, 1, 4), None, None, True, market="stocks"),
    ListQuery(D(2019, 6, 3), None, None, True, market="stocks"),
]


def svc(out: Path, src: FakeSource, logs: list[str]) -> ProbeMassivePaidService:
    return ProbeMassivePaidService(
        src,
        MassivePaidFiles(out),
        calendar=[D(2014, 6, 6), D(2014, 6, 9)],
        tiingo_prices={"MSFT": [ReferenceBar(D(2014, 6, 6), 700.0, 100.0)]},
        tiingo_actions={"MSFT": [ReferenceAction(D(2014, 6, 9), "split", 7.0)]},
        clock=lambda: 0.0,
        progress=logs.append,
        tickers=["MSFT", "GM"],
        list_queries=LISTS,
    )


def test_run_outputs_seal_and_resume(tmp_path: Path) -> None:
    logs: list[str] = []
    r = svc(tmp_path, FakeSource(), logs).run()
    s = r.summary
    assert r.stopped is None and r.remaining == []
    msft = s["prices"]["MSFT"]
    assert msft["raw"]["rows"] == 2 and msft["raw"]["dropped_after_2019"] == 1
    assert "vs_tiingo" in msft and "segments" in s["prices"]["GM"]
    act = s["actions"]["MSFT"]
    assert act["splits"] == 1 and act["dividend_dates_masked"] == 1
    assert act["vs_tiingo"]["splits"]["match"] == 1
    assert act["self_split_adjust_vs_adjusted"]["days_over_1e-6"] == 1  # 假数据 adj 为 0.5 倍
    fill = s["labels"]["fill"]["2010-01-04_active"]
    assert fill["primary_exchange"]["filled_by"]["rule1"] == 1
    assert s["seal"]["price_rows_dropped_after_2019"] == 4
    assert s["seal"]["list_rows_dropped_list_date_after_2019"] == 2
    assert s["seal"]["dividend_dates_masked"] == 2
    text = "".join(p.read_text("utf-8") for p in tmp_path.rglob("*") if p.is_file())
    for w in ("Secret Corp", "2020-", "2021-", KEY):
        assert w not in text, w
    assert all(KEY not in x for x in logs)
    samples = (tmp_path / "samples.csv").read_text("utf-8")
    assert "6770" in samples
    # 续传：再次运行不再发请求
    src2 = FakeSource()
    r2 = svc(tmp_path, src2, []).run()
    assert src2.calls == [] and r2.remaining == []


def test_plan_rejection_stops_and_records(tmp_path: Path) -> None:
    src = FakeSource(reject_prices=True)
    r = svc(tmp_path, src, []).run()
    assert r.stopped == "plan_rejected"
    assert r.summary["plan_rejected"]["http_status"] == 403
    assert "plan does not include history" in r.summary["plan_rejected"]["message"]
    assert src.calls == ["aggs:MSFT:False"]  # 不继续其他请求，不换密钥重试
    src2 = FakeSource()
    r2 = svc(tmp_path, src2, []).run()
    assert r2.stopped == "plan_rejected" and src2.calls == []
