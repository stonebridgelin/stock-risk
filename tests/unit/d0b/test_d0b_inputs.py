"""T12-D0b 第 2.3 节（端到端到规范读取器）与第 2.6 节（完整性、上市期间、日历、下游一致）。

全部用假传输层与临时目录；不读 data/ 下的真实文件，不联网。
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from stock_risk.application.build_canonical import CanonicalBuildError
from stock_risk.application.d0b_checks import (
    InputAction,
    check_actions_consistent,
    check_bars_consistent,
)
from stock_risk.application.fetch_massive_d0b import FetchMassiveD0bService
from stock_risk.application.market_data_loader import MarketDataLoader
from stock_risk.bootstrap import run_build_canonical
from stock_risk.infrastructure.storage.canonical_inputs import read_massive_bars
from stock_risk.infrastructure.storage.canonical_store import (
    CanonicalDataReader,
    read_adjustment_log,
)
from stock_risk.infrastructure.storage.csv_raw_reader import read_manifest
from stock_risk.infrastructure.storage.massive_d0b_files import MassiveD0bFiles
from tests.unit.d0b.fakes import (
    HOST,
    FakeMassive,
    bar_json,
    ok,
    snapshot,
    spec,
    weekdays,
    write_calendar,
    write_config,
)

START, END = date(2008, 1, 1), date(2019, 12, 31)
DAYS = weekdays(date(2015, 3, 2), date(2015, 4, 30))


class Fixed:
    def __init__(self, value) -> None:  # noqa: ANN001
        self.value = value

    def load(self):  # noqa: ANN201
        return self.value


def bars_json(days: list[date], base: float = 100.0) -> str:
    out = []
    for i, d in enumerate(days):
        c = base + (i % 5)
        out.append(bar_json(d, str(c), str(c + 1), str(c - 1), str(c)))
    return ok(out)


def div_json(d: date, amount: str, extra: str = "") -> str:
    return (
        f'{{"ex_dividend_date":"{d.isoformat()}","cash_amount":{amount},"dividend_type":"CD",'
        f'"declaration_date":"2015-02-02"{extra}}}'
    )


def fetch(base: Path, responses: dict, tickers: list[str]) -> Path:
    fake = FakeMassive(responses)
    specs = [spec(t, c, START, END) for t in tickers for c in ("raw", "splits", "dividends")]
    download = base / "download"
    src = fake.source()
    FetchMassiveD0bService(specs, HOST, lambda: MassiveD0bFiles(download, HOST), lambda: src).run()
    return snapshot(download, base / "inputs_v1")


def build(base: Path, inputs: Path, symbols: list[str], cfg: Path | None = None, out: str = "out"):  # noqa: ANN201
    cfg = cfg or write_config(base / "config")
    cal = write_calendar(base / "calendar.csv", weekdays(date(2008, 1, 2), date(2016, 12, 30)))
    return run_build_canonical(
        massive_dir=inputs,
        calendar_file=cal,
        config_dir=cfg,
        out_root=base / out,
        symbols=symbols,
        project_root=base,
        args={},
    )


# ---- 第 2.3 节：端到端 ----


def test_currency_three_cases_end_to_end(tmp_path: Path) -> None:
    ex = DAYS[10]
    responses = {}
    for t, cur in (("USDX", ',"currency":"USD"'), ("CADX", ',"currency":"CAD"'), ("NONX", "")):
        responses[("raw", t)] = bars_json(DAYS)
        responses[("dividends", t)] = ok([div_json(ex, "0.5", cur)])
    inputs = fetch(tmp_path, responses, ["USDX", "CADX", "NONX"])
    with (inputs / "actions/NONX_dividends.csv").open(encoding="utf-8") as f:
        assert next(csv.DictReader(f))["currency"] == ""  # 来源无币种字段：写空
    s = build(tmp_path, inputs, ["USDX", "CADX", "NONX"])
    by = {r.security: r for r in s.results}
    assert by["USDX"].reasons == () and not [
        a for a in by["USDX"].assumptions if a.code == "CURRENCY_ASSUMED_USD"
    ]
    assert by["CADX"].reasons == ("CURRENCY_NOT_USD",)
    assert by["NONX"].reasons == ()
    assert [a.code for a in by["NONX"].assumptions] == ["CURRENCY_ASSUMED_USD"]


def test_raw_number_text_end_to_end_to_reader(tmp_path: Path) -> None:
    days = DAYS[:3]
    raw = ok(
        [
            bar_json(days[0], "1.00000050000000001", "1.1", "0.9", "1.00000050000000001", "2.5E+3"),
            bar_json(days[1], "1", "1.1", "0.9", "1", "1e-7"),
            bar_json(days[2], "1", "1.1", "0.9", "1", "1000"),
        ]
    )
    inputs = fetch(tmp_path, {("raw", "NUM"): raw}, ["NUM"])
    got = read_massive_bars(inputs, "NUM")
    assert got.rows[0].open == "1.00000050000000001" and got.rows[0].volume == "2.5E+3"
    assert got.rows[1].volume == "1e-7"
    s = build(tmp_path, inputs, ["NUM"])
    (r,) = s.results
    # 原文 1.00000050000000001 量化为 1.000001；若先变成 float（1.0000005）会量化为 1.000000
    assert str(r.bars[0].f_open) == "1.000001"
    assert str(r.bars[0].volume) == "2.5E+3"


def test_masked_declaration_is_decl_not_prior(tmp_path: Path) -> None:
    ex = DAYS[10]
    dividend = (
        f'{{"ex_dividend_date":"{ex.isoformat()}","cash_amount":0.5,"dividend_type":"CD",'
        '"declaration_date":"2020-01-05","pay_date":"2020-01-20","currency":"USD"}'
    )
    inputs = fetch(
        tmp_path, {("raw", "MSK"): bars_json(DAYS), ("dividends", "MSK"): ok([dividend])}, ["MSK"]
    )
    with (inputs / "actions/MSK_dividends.csv").open(encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["declaration_date"] == "" and row["masked_after_2019"] == "declaration_date|pay_date"
    (r,) = build(tmp_path, inputs, ["MSK"]).results
    (rec,) = r.actions
    assert "DECL_NOT_PRIOR" in rec.codes and "DECL_DATE_MISSING" not in rec.codes


# ---- 第 2.6 节第 1 项：完整性 ----


def _inputs_with(tmp_path: Path) -> Path:
    return fetch(tmp_path, {("raw", "AAA"): bars_json(DAYS)}, ["AAA"])


def _rewrite_status(inputs: Path, rel: str, **changes: object) -> None:
    p = inputs / "INPUTS_STATUS.json"
    data = json.loads(p.read_text("utf-8"))
    data["files"][rel].update(changes)
    p.write_text(json.dumps(data), "utf-8")


def test_incomplete_action_file_stops_even_with_right_hash(tmp_path: Path) -> None:
    inputs = _inputs_with(tmp_path)
    _rewrite_status(inputs, "actions/AAA_splits.csv", status="incomplete", complete=False)
    with pytest.raises(CanonicalBuildError, match="成功且完整"):
        build(tmp_path, inputs, ["AAA"])
    assert not (tmp_path / "out").exists()


def test_missing_action_file_stops(tmp_path: Path) -> None:
    inputs = _inputs_with(tmp_path)
    (inputs / "actions/AAA_dividends.csv").unlink()
    with pytest.raises(CanonicalBuildError, match="文件缺失"):
        build(tmp_path, inputs, ["AAA"])
    inputs2 = _inputs_with(tmp_path / "b")
    data = json.loads((inputs2 / "INPUTS_STATUS.json").read_text("utf-8"))
    del data["files"]["actions/AAA_splits.csv"]
    (inputs2 / "INPUTS_STATUS.json").write_text(json.dumps(data), "utf-8")
    with pytest.raises(CanonicalBuildError, match="没有登记"):
        build(tmp_path / "b", inputs2, ["AAA"])


def test_legal_empty_actions_build_normally(tmp_path: Path) -> None:
    inputs = _inputs_with(tmp_path)
    (r,) = build(tmp_path, inputs, ["AAA"]).results
    assert r.actions == () and r.eligibility.value == "USABLE"


def test_hash_mismatch_stops(tmp_path: Path) -> None:
    inputs = _inputs_with(tmp_path)
    _rewrite_status(inputs, "aggs/AAA_raw.csv", sha256="0" * 64)
    with pytest.raises(CanonicalBuildError, match="哈希"):
        build(tmp_path, inputs, ["AAA"])


def test_missing_inputs_status_stops(tmp_path: Path) -> None:
    inputs = _inputs_with(tmp_path)
    (inputs / "INPUTS_STATUS.json").unlink()
    with pytest.raises(CanonicalBuildError, match="INPUTS_STATUS"):
        build(tmp_path, inputs, ["AAA"])


# ---- 第 2.6 节第 2 项：上市期间截取 ----


def test_listing_clip_and_dataset_keys(tmp_path: Path) -> None:
    hist_days = [date(2008, 1, 2), date(2008, 1, 3), *DAYS]
    responses = {
        ("raw", "LST"): bars_json(hist_days),
        ("splits", "LST"): ok(
            [
                # 区间之前：截掉并计数
                '{"execution_date":"1987-05-12","split_from":1,"split_to":2}',
                # 上市止日之后
                f'{{"execution_date":"{DAYS[-1].isoformat()}","split_from":1,"split_to":2}}',
            ]
        ),
        ("dividends", "LST"): ok(
            [
                # 2008-01-01 节假日除息、锚日 2008-01-02：区间之内，保留，按锚日行动处理
                '{"ex_dividend_date":"2008-01-01","cash_amount":0.1,"dividend_type":"CD","currency":"USD"}',
            ]
        ),
    }
    inputs = fetch(tmp_path, responses, ["LST"])
    until = DAYS[-3]
    cfg = write_config(
        tmp_path / "config",
        listing_rows=[f"LST,,{until.isoformat()},NYSE,https://example.invalid/8k,2026-10-18,测试"],
    )
    s = build(tmp_path, inputs, ["LST"], cfg)
    (r,) = s.results
    assert r.bars[-1].day == until
    dataset = json.loads((s.out_dir / "_dataset.json").read_text("utf-8"))
    assert dataset["securities"]["LST"]["listing_clipped"] == {
        "bars": 2,
        "splits": 2,
        "dividends": 0,
    }
    config = dataset["config"]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    assert config["listing_periods_sha256"] == sha(cfg / "listing_periods.csv")
    assert config["inputs_status_sha256"] == sha(inputs / "INPUTS_STATUS.json")
    assert config["massive_source_tickers_sha256"] == sha(cfg / "massive_source_tickers.csv")
    (g,) = r.groups
    assert g.status.value == "ANCHOR_NOT_APPLIED" and g.original_ex_date == date(2008, 1, 1)


def test_massive_source_ticker_mapping_used(tmp_path: Path) -> None:
    inputs = fetch(tmp_path, {("raw", "SNEX"): bars_json(DAYS)}, ["SNEX"])
    cfg = write_config(tmp_path / "config", massive_rows=["SONX,SNEX,,,PM-T,测试"])
    s = build(tmp_path, inputs, ["SONX"], cfg)
    (r,) = s.results
    assert r.security == "SONX" and len(r.bars) == len(DAYS)
    dataset = json.loads((s.out_dir / "_dataset.json").read_text("utf-8"))
    seg = dataset["securities"]["SONX"]["segments"][0]
    assert seg["source_ticker"] == "SNEX" and seg["bars"]["source"] == "massive:aggs/SNEX_raw.csv"


# ---- 第 2.6 节第 3 项：日历；第 2 项：下游一致 ----


def test_calendar_in_version_and_downstream_consistency(tmp_path: Path) -> None:
    ex = DAYS[10]
    responses = {
        ("raw", "AAA"): bars_json([d for d in DAYS if d != DAYS[5]]),  # 一天缺价
        ("splits", "AAA"): ok(
            [f'{{"execution_date":"{DAYS[20].isoformat()}","split_from":1,"split_to":2}}']
        ),
        ("dividends", "AAA"): ok([div_json(ex, "0.5", ',"currency":"USD"')]),
    }
    inputs = fetch(tmp_path, responses, ["AAA"])
    s = build(tmp_path, inputs, ["AAA"])
    out = s.out_dir
    assert "_calendar.csv" in read_manifest(out / "MANIFEST.sha256")
    reader = CanonicalDataReader(out)
    assert reader.calendar_days()[0] == date(2008, 1, 2)
    assert not (out / "SPY.csv").exists()  # 不伪造 SPY
    loader = MarketDataLoader(reader, Fixed({}), Fixed({}))
    assert loader.calendar().days == reader.calendar_days()
    assert check_bars_consistent(loader, reader, "AAA") == []
    log = read_adjustment_log(out, "AAA")
    (r,) = s.results
    actions = [InputAction(a.action_id, a.original_ex_date) for a in r.actions]
    observed = loader.load("AAA")
    problems = check_actions_consistent(
        "AAA", actions, log, observed.ranges.observation_start, r.bars[-1].day
    )
    assert problems == []
    assert check_actions_consistent(
        "AAA", actions[:1], log, DAYS[0], r.bars[-1].day
    )  # 少一条即报告
