"""T12-D0b 第 2.3 节：补取工具（只用假传输层，不发任何真实请求）。"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import pytest

from stock_risk.application.fetch_massive_d0b import FetchMassiveD0bService
from stock_risk.domain.seal import SealViolation
from stock_risk.infrastructure.storage.massive_d0b_files import MassiveD0bFiles
from stock_risk.infrastructure.storage.massive_source_host import SourceHostConflict
from tests.unit.d0b.fakes import HOST, FakeMassive, bar_json, ok, spec

D1, D2 = date(2015, 3, 2), date(2015, 3, 3)
START, END = date(2008, 1, 1), date(2019, 12, 31)


def service(fake: FakeMassive, out: Path, specs, host: str = HOST) -> FetchMassiveD0bService:  # noqa: ANN001
    src = fake.source()
    return FetchMassiveD0bService(specs, host, lambda: MassiveD0bFiles(out, HOST), lambda: src)


def ledger(out: Path) -> dict:
    return json.loads((out / "_ledger.json").read_text("utf-8"))["requests"]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_only_gaps_requested_and_resume(tmp_path: Path) -> None:
    fake = FakeMassive({("raw", "AAA"): ok([bar_json(D1, "10", "11", "9", "10")])})
    specs = [spec("AAA", "raw", START, END), spec("AAA", "splits", START, END)]
    r = service(fake, tmp_path, specs).run()
    assert sorted(fake.calls) == [("raw", "AAA"), ("splits", "AAA")]  # 只请求缺口中的两项
    assert len(r.completed) == 2 and r.requests_now == 2
    entries = ledger(tmp_path)
    raw = entries[f"{HOST}|raw|AAA|2008-01-01|2019-12-31|false"]
    assert raw["status"] == "ok" and raw["complete"] is True and raw["pages"] == 1
    assert raw["rows"] == 1 and raw["file"] == "aggs/AAA_raw.csv"
    split = entries[f"{HOST}|splits|AAA||2019-12-31|"]  # 行动请求没有下限
    assert split["rows"] == 0 and (tmp_path / "actions/AAA_splits.csv").read_text(
        "utf-8"
    ).startswith("execution_date,split_from,split_to,source_host")
    fake.calls.clear()
    svc = service(fake, tmp_path, specs)
    assert len(svc.plan().done) == 2 and svc.plan().pending == ()
    r2 = svc.run()
    assert fake.calls == [] and r2.requests_now == 0  # 已完成的不重发


def test_request_end_after_2019_refused_before_any_request(tmp_path: Path) -> None:
    fake = FakeMassive({})
    with pytest.raises(SealViolation):
        service(fake, tmp_path, [spec("AAA", "raw", START, date(2020, 1, 2))])
    with pytest.raises(SealViolation):
        service(fake, tmp_path, [spec("AAA", "dividends", START, date(2020, 1, 2))])
    assert fake.calls == []


def test_sealed_rows_in_response_filtered_and_counted(tmp_path: Path) -> None:
    fake = FakeMassive(
        {
            ("raw", "AAA"): ok(
                [
                    bar_json(D1, "10", "11", "9", "10"),
                    bar_json(date(2020, 1, 2), "1", "1", "1", "1"),
                ]
            ),
            ("splits", "AAA"): ok(['{"execution_date":"2020-02-03","split_from":1,"split_to":2}']),
            ("dividends", "AAA"): ok(
                ['{"ex_dividend_date":"2020-01-06","cash_amount":0.5,"dividend_type":"CD"}']
            ),
        }
    )
    specs = [spec("AAA", c, START, END) for c in ("raw", "splits", "dividends")]
    service(fake, tmp_path, specs).run()
    entries = ledger(tmp_path)
    assert [e["sealed_dropped"] for e in entries.values()] == [1, 1, 1]
    assert [r["date"] for r in rows(tmp_path / "aggs/AAA_raw.csv")] == ["2015-03-02"]
    assert rows(tmp_path / "actions/AAA_splits.csv") == []
    assert rows(tmp_path / "actions/AAA_dividends.csv") == []


def test_resume_after_partial_failures(tmp_path: Path) -> None:
    # 原始日线完成而复权日线未完成（翻页超限）；拆股完成而分红失败（HTTP 500）
    fake = FakeMassive(
        {
            ("raw", "AAA"): ok([bar_json(D1, "10", "11", "9", "10")]),
            ("adj", "AAA"): '{"status":"OK","results":[],"next_url":"https://relaynex.vip/v2/x"}',
            ("dividends", "AAA"): 500,
        }
    )
    specs = [spec("AAA", c, START, END) for c in ("raw", "adj", "splits", "dividends")]
    src = fake.source()

    def svc() -> FetchMassiveD0bService:
        return FetchMassiveD0bService(
            specs, HOST, lambda: MassiveD0bFiles(tmp_path, HOST), lambda: src
        )

    import stock_risk.infrastructure.massive.client as client

    old = client.MAX_PAGES
    client.MAX_PAGES = 1  # 让复权日线在第 1 页之后仍有下一页 → 未完成
    try:
        r = svc().run()
    finally:
        client.MAX_PAGES = old
    assert len(r.completed) == 2 and len(r.incomplete) == 1 and len(r.failed) == 1
    entries = ledger(tmp_path)
    assert entries[f"{HOST}|adj|AAA|2008-01-01|2019-12-31|true"]["status"] == "incomplete"
    assert entries[f"{HOST}|dividends|AAA||2019-12-31|"]["status"] == "failed"
    # 第二次：只重发未完成与失败的两项
    fake.responses[("adj", "AAA")] = ok([bar_json(D1, "5", "6", "4", "5")])
    fake.responses[("dividends", "AAA")] = ok([])
    fake.calls.clear()
    r2 = svc().run()
    assert sorted(fake.calls) == [("adj", "AAA"), ("dividends", "AAA")]
    assert len(r2.completed) == 2 and not r2.failed and not r2.incomplete
    assert (tmp_path / "vendor_adj/AAA_adj.csv").exists()


def test_auth_error_stops_immediately(tmp_path: Path) -> None:
    fake = FakeMassive({("raw", "AAA"): 403})
    specs = [spec("AAA", "raw", START, END), spec("BBB", "raw", START, END)]
    r = service(fake, tmp_path, specs).run()
    assert r.stopped == "HTTP 403" and r.remaining == 1
    assert fake.calls == [("raw", "AAA")]


def test_wrong_host_refused(tmp_path: Path) -> None:
    fake = FakeMassive({})
    with pytest.raises(ValueError, match="只经中转服务"):
        service(fake, tmp_path, [spec("AAA", "raw", START, END)], host="api.massive.com")
    (tmp_path / "_ledger.json").write_text(
        '{"source_host":"api.massive.com","requests":{}}', "utf-8"
    )
    with pytest.raises(SourceHostConflict):
        MassiveD0bFiles(tmp_path, HOST)


def test_dry_run_plan_needs_no_source(tmp_path: Path) -> None:
    specs = [spec("AAA", c, START, END) for c in ("raw", "adj", "splits", "dividends")]
    svc = FetchMassiveD0bService(specs, HOST, lambda: MassiveD0bFiles(tmp_path / "d", HOST), None)
    plan = svc.plan()
    assert len(plan.pending) == 4 and plan.counts() == {
        "raw": 1,
        "adj": 1,
        "splits": 1,
        "dividends": 1,
    }
    assert not (tmp_path / "d").exists()  # 不写任何文件
    with pytest.raises(RuntimeError, match="dry-run"):
        svc.run()


def test_raw_number_text_and_masking_in_files(tmp_path: Path) -> None:
    fake = FakeMassive(
        {
            ("raw", "AAA"): ok(
                [
                    bar_json(
                        D1, "1.00000050000000001", "2.5E+3", "1e-7", "1.00000050000000001", "2.5E+3"
                    )
                ]
            ),
            ("dividends", "AAA"): ok(
                [
                    '{"ex_dividend_date":"2019-12-20","cash_amount":1e-7,"dividend_type":"CD",'
                    '"declaration_date":"2019-11-01","record_date":"2019-12-23",'
                    '"pay_date":"2020-01-10","currency":"USD"}'
                ]
            ),
        }
    )
    service(fake, tmp_path, [spec("AAA", c, START, END) for c in ("raw", "dividends")]).run()
    (bar,) = rows(tmp_path / "aggs/AAA_raw.csv")
    assert (bar["open"], bar["high"], bar["low"], bar["volume"]) == (
        "1.00000050000000001",
        "2.5E+3",
        "1e-7",
        "2.5E+3",
    )
    (d,) = rows(tmp_path / "actions/AAA_dividends.csv")
    assert (
        d["cash_amount"] == "1e-7" and d["pay_date"] == "" and d["masked_after_2019"] == "pay_date"
    )
    assert d["record_date"] == "2019-12-23" and d["currency"] == "USD"
    entry = ledger(tmp_path)[f"{HOST}|dividends|AAA||2019-12-31|"]
    assert entry["dates_masked"] == 1 and entry["currency_field_missing"] == 0
