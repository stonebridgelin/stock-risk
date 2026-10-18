"""T12-D0b P1 验收补正（docs/reviews/2026-10-18_T12-D0b_P1验收.md 第 3 节）：P1-01 至 P1-03。

只用合成数据与假传输层；不读 data/ 下的真实文件，不联网。
"""

from __future__ import annotations

import csv
import json
import re
import urllib.error
import urllib.request
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

import stock_risk.bootstrap as bootstrap
import stock_risk.infrastructure.massive.client as client
from stock_risk.application.build_canonical import CanonicalBuildError
from stock_risk.application.fetch_massive_d0b import FetchMassiveD0bService
from stock_risk.application.ports import QueryRejected
from stock_risk.config.settings import Secret
from stock_risk.infrastructure.massive.client import (
    RELAY_BASE_URL,
    MassiveReference,
    parse_json_keep_numbers,
)
from stock_risk.infrastructure.storage.canonical_inputs import MassiveCanonicalSource
from stock_risk.infrastructure.storage.massive_d0b_files import MassiveD0bFiles
from stock_risk.interfaces.cli.main import main
from tests.unit.d0b.fakes import HOST, FakeMassive, bar_json, ok, snapshot, spec

START, END = date(2008, 1, 1), date(2019, 12, 31)
D1 = date(2015, 3, 2)


def _ledger(out: Path) -> dict[str, Any]:
    return json.loads((out / "_ledger.json").read_text("utf-8"))["requests"]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---- P1-01：缺日期的行动单独计数，不当作封存，不静默丢弃 ----


def test_missing_dates_counted_separately_and_reported(tmp_path: Path) -> None:
    fake = FakeMassive(
        {
            ("splits", "AAA"): ok(
                [
                    '{"execution_date":"2015-03-03","split_from":1,"split_to":2}',
                    '{"split_from":1,"split_to":3}',  # 缺执行日
                ]
            ),
            ("dividends", "AAA"): ok(
                [
                    '{"ex_dividend_date":"2015-03-10","cash_amount":0.5,"dividend_type":"CD","currency":"USD"}',
                    '{"ex_dividend_date":null,"cash_amount":0.7,"dividend_type":"CD","currency":"USD"}',
                ]
            ),
        }
    )
    src = fake.source()
    specs = [spec("AAA", c, START, END) for c in ("splits", "dividends")]
    r = FetchMassiveD0bService(
        specs, HOST, lambda: MassiveD0bFiles(tmp_path, HOST), lambda: src
    ).run()
    entries = _ledger(tmp_path)
    split = entries[f"{HOST}|splits|AAA||2019-12-31|"]
    div = entries[f"{HOST}|dividends|AAA||2019-12-31|"]
    for e in (split, div):
        assert e["sealed_dropped"] == 0  # 不计入封存数
        assert e["date_missing"] == 1
        assert e["rows"] == 1  # 数据文件只含有日期的那一条
    assert split["date_missing_rows"] == [{"split_from": "1", "split_to": "3"}]
    assert div["date_missing_rows"][0]["cash_amount"] == "0.7"
    assert sorted(r.date_missing) == sorted(
        [f"{HOST}|splits|AAA||2019-12-31|", f"{HOST}|dividends|AAA||2019-12-31|"]
    )  # 问题被报告


def test_missing_date_bar_counted_not_sealed(tmp_path: Path) -> None:
    fake = FakeMassive(
        {
            ("raw", "AAA"): ok(
                [bar_json(D1, "10", "11", "9", "10"), '{"o":1,"h":1,"l":1,"c":1,"v":1}']
            )
        }
    )
    src = fake.source()
    FetchMassiveD0bService(
        [spec("AAA", "raw", START, END)], HOST, lambda: MassiveD0bFiles(tmp_path, HOST), lambda: src
    ).run()
    (e,) = _ledger(tmp_path).values()
    assert (e["sealed_dropped"], e["date_missing"], e["rows"]) == (0, 1, 1)


def test_build_stops_on_unresolved_missing_dates(tmp_path: Path) -> None:
    fake = FakeMassive(
        {
            ("raw", "AAA"): ok([bar_json(D1, "10", "11", "9", "10")]),
            ("splits", "AAA"): ok(['{"split_from":1,"split_to":2}']),
        }
    )
    src = fake.source()
    specs = [spec("AAA", c, START, END) for c in ("raw", "splits", "dividends")]
    download = tmp_path / "download"
    FetchMassiveD0bService(specs, HOST, lambda: MassiveD0bFiles(download, HOST), lambda: src).run()
    inputs = snapshot(download, tmp_path / "inputs")
    status = json.loads((inputs / "INPUTS_STATUS.json").read_text("utf-8"))
    status["files"]["actions/AAA_splits.csv"]["date_missing"] = _ledger(download)[
        f"{HOST}|splits|AAA||2019-12-31|"
    ]["date_missing"]
    (inputs / "INPUTS_STATUS.json").write_text(json.dumps(status), "utf-8")
    with pytest.raises(CanonicalBuildError, match="缺日期"):
        MassiveCanonicalSource(inputs).splits("AAA")


# ---- P1-02：密钥不外泄 ----

SECRET = "SECRETKEY-T12-ZZ7Q-0042"


def _transport(status: int | None):  # noqa: ANN202
    """status 为 None：成功；否则返回该 HTTP 状态，错误正文中故意回显密钥。"""

    def transport(req: urllib.request.Request) -> Any:
        assert req.get_header("Authorization") == f"Bearer {SECRET}"  # 密钥确实只在请求头中
        assert SECRET not in req.full_url
        if status is not None:
            body = json.dumps({"status": "ERROR", "message": f"bad key {SECRET}"}).encode()
            raise urllib.error.HTTPError(req.full_url, status, "x", {}, BytesIO(body))  # type: ignore[arg-type]
        if "/v2/aggs/" in req.full_url:
            return parse_json_keep_numbers(ok([bar_json(D1, "10", "11", "9", "10")]))
        return parse_json_keep_numbers(ok([]))

    return transport


def _files_text(root: Path) -> str:
    return "".join(
        p.read_bytes().decode("utf-8", "replace") for p in sorted(root.rglob("*")) if p.is_file()
    )


@pytest.mark.parametrize("status", [None, 401, 403, 500])
def test_secret_never_leaks(tmp_path: Path, monkeypatch, capsys, status: int | None) -> None:  # noqa: ANN001
    from tests.unit.d0b.test_d0b_inventory import _files

    identity, gaps = _files(tmp_path)

    class S:
        massive_advanced_base_url = "https://relaynex.vip"
        massive_advanced_api_key = Secret(SECRET)
        massive_per_minute = 600

    monkeypatch.setattr(bootstrap, "load_settings", lambda: S())
    monkeypatch.setattr(bootstrap, "make_transport", lambda parse: _transport(status))
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    monkeypatch.setattr(client.time, "sleep", lambda _: None)
    out = tmp_path / "download"
    code = main(
        ["fetch-massive-d0b", "--identity", str(identity), "--gaps", str(gaps), "--out", str(out)]
    )
    captured = capsys.readouterr()
    assert code == (0 if status is None else 1)
    assert SECRET not in captured.out and SECRET not in captured.err
    assert SECRET not in _files_text(out)  # 台账与全部数据文件
    # 异常信息
    source = MassiveReference(
        SECRET,
        base_url=RELAY_BASE_URL,
        per_minute=600,
        transport=_transport(status),
        sleep=lambda _: None,
    )
    if status is None:
        assert source.aggs("AAA", START, END, False) is not None
    else:
        with pytest.raises(QueryRejected) as exc:
            source.aggs("AAA", START, END, False)
        assert SECRET not in str(exc.value) and SECRET not in repr(exc.value)
        assert "***" in str(exc.value)  # 回显的密钥被替换


# ---- P1-03：币种为 null 计入缺失 ----


def test_currency_null_counted_as_missing(tmp_path: Path) -> None:
    fake = FakeMassive(
        {
            ("dividends", "AAA"): ok(
                [
                    '{"ex_dividend_date":"2015-03-10","cash_amount":0.5,"dividend_type":"CD","currency":null}',
                    '{"ex_dividend_date":"2015-06-10","cash_amount":0.5,"dividend_type":"CD","currency":"USD"}',
                ]
            )
        }
    )
    src = fake.source()
    FetchMassiveD0bService(
        [spec("AAA", "dividends", START, END)],
        HOST,
        lambda: MassiveD0bFiles(tmp_path, HOST),
        lambda: src,
    ).run()
    (e,) = _ledger(tmp_path).values()
    assert e["currency_field_missing"] == 1
    assert [r["currency"] for r in _rows(tmp_path / "actions/AAA_dividends.csv")] == ["", "USD"]


# ---- P1-05：缺日期记录的原文先遮蔽封存期日期，再写入台账 ----


def test_missing_date_rows_mask_sealed_dates_in_ledger(tmp_path: Path) -> None:
    fake = FakeMassive(
        {
            # 分红缺除息日，派息日 2020-01-10、公告日 2020-01-02（验收记录中的例子）
            ("dividends", "AAA"): ok(
                [
                    '{"cash_amount":0.7,"dividend_type":"CD","currency":"USD",'
                    '"declaration_date":"2020-01-02","record_date":"2019-12-27",'
                    '"pay_date":"2020-01-10"}'
                ]
            ),
            # 拆股缺执行日，另带 2020 年的日期字段
            ("splits", "AAA"): ok(['{"split_from":1,"split_to":2,"announced_date":"2020-02-03"}']),
        }
    )
    src = fake.source()
    specs = [spec("AAA", c, START, END) for c in ("splits", "dividends")]
    FetchMassiveD0bService(specs, HOST, lambda: MassiveD0bFiles(tmp_path, HOST), lambda: src).run()
    text = (tmp_path / "_ledger.json").read_text("utf-8")
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", text)
    assert dates  # 台账中确有日期（请求止日、登记日），检查不是空转
    assert all(d <= "2019-12-31" for d in dates), dates  # 没有任何晚于 2019-12-31 的日期
    entries = _ledger(tmp_path)
    (div_row,) = entries[f"{HOST}|dividends|AAA||2019-12-31|"]["date_missing_rows"]
    assert div_row["pay_date"] == "" and div_row["declaration_date"] == ""
    assert div_row["record_date"] == "2019-12-27"  # 不晚于截止日的照留
    assert set(div_row["masked_after_2019"].split("|")) == {"declaration_date", "pay_date"}
    (split_row,) = entries[f"{HOST}|splits|AAA||2019-12-31|"]["date_missing_rows"]
    assert split_row["announced_date"] == "" and split_row["masked_after_2019"] == "announced_date"
