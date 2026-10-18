"""规范数据版本的构建与读取（T11-D0a 第 7 节）。

B13–B16、B21、B22、B24、B26，以及窗口集合（第 4.7 节）。

全部在 pytest 的临时目录上运行：输入为测试中构造的 Massive 格式文件，不读 data/ 下的任何真实文件。
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from stock_risk.application.build_canonical import (
    CanonicalBuildError,
    CanonicalConfig,
    atr_windows,
    build_canonical,
    default_params,
)
from stock_risk.application.market_data_loader import MarketDataLoader
from stock_risk.bootstrap import run_build_canonical
from stock_risk.domain.audit import CanonicalBarPrices, canonical_factor_checks
from stock_risk.domain.corrections import CorrectionError, CorrectionMethod, PriceCorrection
from stock_risk.domain.evaluation.params import ATR_PERIOD
from stock_risk.domain.lifecycle.params import GRID_DIMENSIONS, REFERENCE_NUMERIC
from stock_risk.domain.seal import SealViolation
from stock_risk.infrastructure.storage.canonical_inputs import MassiveCanonicalSource, read_calendar
from stock_risk.infrastructure.storage.canonical_store import (
    BUILD_META_FILE,
    CanonicalDataReader,
    manifest_text,
    open_market_reader,
)
from stock_risk.infrastructure.storage.csv_raw_reader import (
    CanonicalModeError,
    CsvRawDataReader,
    read_manifest,
)
from stock_risk.infrastructure.storage.csv_raw_repository import BAR_COLUMNS
from tests.unit.adjustment.helpers import weekdays

DAYS = weekdays(date(2015, 3, 2), date(2015, 4, 30))
CLOCK = iter(datetime(2026, 10, 18, 1, 0, i, tzinfo=UTC) for i in range(10_000))
HOST = "test.invalid"


class Fixed:
    def __init__(self, value) -> None:  # noqa: ANN001
        self.value = value

    def load(self):  # noqa: ANN201
        return self.value


def _csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def write_ticker(
    massive: Path,
    ticker: str,
    closes: dict[date, str | tuple[str, str, str, str]],
    splits: list[list[object]] = (),  # type: ignore[assignment]
    dividends: list[list[object]] = (),  # type: ignore[assignment]
) -> None:
    """只给收盘的：开 = 收，高 = 收 + 1，低 = 收 − 1（避免 ATR 为 0）；元组为 (开, 高, 低, 收)。"""
    rows = []
    for d, c in sorted(closes.items()):
        if isinstance(c, str):
            c = (c, str(Decimal(c) + 1), str(Decimal(c) - 1), c)
        rows.append([d.isoformat(), *c, "1000", "", "", HOST])
    _csv(
        massive / "aggs" / f"{ticker}_raw.csv",
        ["date", "open", "high", "low", "close", "volume", "vwap", "trades", "source_host"],
        rows,
    )
    _csv(
        massive / "actions" / f"{ticker}_splits.csv",
        ["execution_date", "split_from", "split_to", "source_host"],
        [[*s, HOST] for s in splits],
    )
    _csv(
        massive / "actions" / f"{ticker}_dividends.csv",
        [
            "ex_dividend_date",
            "cash_amount",
            "dividend_type",
            "declaration_date",
            "record_date",
            "pay_date",
            "masked_after_2019",
            "source_host",
        ],
        [[*d, HOST] for d in dividends],
    )


def write_config(cfg: Path) -> None:
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "symbol_segments.csv").write_text("symbol,source_ticker,from,to,note\n", "utf-8")
    (cfg / "adjustment_verifications.csv").write_text(
        "id,target_type,target,source_sha256,kind,value,ruling,evidence\n", "utf-8"
    )
    (cfg / "raw_price_corrections.csv").write_text(
        "id,security,day,field,value,ruling,evidence,version\n", "utf-8"
    )
    (cfg / "price_corrections.csv").write_text(
        "symbol,from,to,method,ruling,evidence,d0a_status\n", "utf-8"
    )


def make_inputs(base: Path) -> tuple[Path, Path, Path]:
    """SPY（无行动）、AAA（拆股 + 两笔同日分红，A12 形态）、BBB（异常分红，不可正式使用）。"""
    massive, cfg = base / "massive", base / "config"
    write_ticker(massive, "SPY", {d: "200" for d in DAYS})
    aaa = (
        {d: "100" for d in DAYS[:10]}
        | {d: "97" for d in DAYS[10:20]}
        | {d: "48.5" for d in DAYS[20:]}
    )
    write_ticker(
        massive,
        "AAA",
        aaa,
        splits=[[DAYS[20].isoformat(), "1", "2"]],
        dividends=[
            [DAYS[10].isoformat(), "1", "CD", "2015-01-02", "", "", ""],
            [DAYS[10].isoformat(), "2", "CD", "2015-01-02", "", "", ""],
        ],
    )
    write_ticker(
        massive,
        "BBB",
        {d: "10" for d in DAYS[:5]} | {d: "4" for d in DAYS[5:]},
        dividends=[[DAYS[5].isoformat(), "6", "CD", "", "", "", ""]],
    )
    cal = base / "calendar.csv"
    _csv(cal, ["date"], [[d.isoformat()] for d in DAYS])
    write_config(cfg)
    return massive, cal, cfg


def write_status(massive: Path) -> None:
    """输入快照的 INPUTS_STATUS.json（T12-D0b 第 2.6 节第 1 项）。

    目录中的每个输入文件都登记为成功且完整。"""
    files = {
        p.relative_to(massive).as_posix(): {
            "status": "ok",
            "complete": True,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted([*massive.glob("aggs/*_raw.csv"), *massive.glob("actions/*.csv")])
    }
    (massive / "INPUTS_STATUS.json").write_text(json.dumps({"files": files}), "utf-8")


def build(base: Path, out: Path, symbols=("SPY", "AAA", "BBB")):  # noqa: ANN001, ANN201
    massive, cal, cfg = base / "massive", base / "calendar.csv", base / "config"
    write_status(massive)
    return run_build_canonical(
        massive_dir=massive,
        calendar_file=cal,
        config_dir=cfg,
        out_root=out,
        symbols=list(symbols),
        project_root=base,  # 临时目录不是 git 仓库：提交号为空
        args={"test": True},
    )


@pytest.fixture
def built(tmp_path: Path):  # noqa: ANN201
    make_inputs(tmp_path)
    summary = build(tmp_path, tmp_path / "canonical")
    return tmp_path, summary


def files_of(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# ---- 第 4.7 节：窗口集合取自配置 ----


def test_atr_windows_come_from_grid_reference_and_evaluation() -> None:
    expected = set(dict(GRID_DIMENSIONS)["n"]) | {REFERENCE_NUMERIC["n"], ATR_PERIOD}
    assert set(atr_windows()) == expected and atr_windows() == tuple(sorted(expected))
    assert default_params().gap_breaks == (10, 20)


# ---- 目录、报告与 B13、B14 ----


def test_build_writes_reports(built) -> None:  # noqa: ANN001
    base, s = built
    out = s.out_dir
    assert out.name == s.version and s.version.startswith("d0a-") and len(s.version) == 16
    assert s.total == 3 and s.usable == 2 and s.not_usable_by_reason == {"ABNORMAL_FACTOR": 1}
    names = set(files_of(out))
    assert {"SPY.csv", "AAA.csv", "diagnostic/BBB.csv", "_eligibility.csv"} <= names
    assert {f"{t}.adjlog.json" for t in ("SPY", "AAA", "BBB")} <= names
    assert "BBB.csv" not in names
    with (out / "_eligibility.csv").open(encoding="utf-8") as f:
        rows = {r["security"]: r for r in csv.DictReader(f)}
    assert set(rows) == {"SPY", "AAA", "BBB"}  # 不可正式使用的也在清单中
    assert (
        rows["BBB"]["eligibility"] == "NOT_USABLE" and rows["BBB"]["reasons"] == "ABNORMAL_FACTOR"
    )
    diag = (out / "diagnostic" / "BBB.csv").read_text("utf-8").splitlines()
    assert diag[0].startswith("status,") and all(ln.startswith("复权未完成,") for ln in diag[1:])
    header = (out / "AAA.csv").read_text("utf-8").splitlines()[0]
    assert header.split(",") == BAR_COLUMNS
    manifest = read_manifest(out / "MANIFEST.sha256")
    assert BUILD_META_FILE not in manifest and "MANIFEST.sha256" not in manifest
    assert "_dataset.json" in manifest and "diagnostic/BBB.csv" in manifest
    dataset = json.loads((out / "_dataset.json").read_text("utf-8"))
    assert "version" not in dataset and dataset["spec_version"] == "D0a-v4"
    assert dataset["params"]["gap_breaks"] == [10, 20]
    assert dataset["params"]["atr_windows"] == [14, 20]
    assert dataset["securities"]["AAA"]["anchor"] == DAYS[0].isoformat()


def test_b13_existing_output_dir_refused(built) -> None:  # noqa: ANN001
    base, s = built
    before = files_of(base / "canonical")
    with pytest.raises(CanonicalBuildError, match="已存在"):
        build(base, base / "canonical")
    assert files_of(base / "canonical") == before  # 不写任何文件


def test_b14_same_input_byte_identical(built) -> None:  # noqa: ANN001
    base, s1 = built
    s2 = build(base, base / "canonical2")
    assert s1.version == s2.version
    a, b = files_of(s1.out_dir), files_of(s2.out_dir)
    a.pop(BUILD_META_FILE)
    b.pop(BUILD_META_FILE)
    assert a == b


# ---- B15：规范数据与 use_raw ----


def _use_raw(status: str | None) -> dict:
    c = PriceCorrection(
        "AAA", DAYS[2], DAYS[4], CorrectionMethod.USE_RAW, "PM-T", "测试", d0a_status=status
    )
    return {"AAA": (c,)}


def test_b15_unabsorbed_use_raw_raises_on_all_entries(built) -> None:  # noqa: ANN001
    _, s = built
    loader = MarketDataLoader(CanonicalDataReader(s.out_dir), Fixed({}), Fixed(_use_raw(None)))
    for call in (lambda: loader.load("AAA"), lambda: loader.load("AAA", corrected=False)):
        with pytest.raises(CorrectionError, match="absorbed"):
            call()
    with pytest.raises(CorrectionError, match="absorbed"):
        loader.audit_view("AAA")


def test_b15_absorbed_use_raw_not_applied_to_f(built) -> None:  # noqa: ANN001
    _, s = built
    loader = MarketDataLoader(
        CanonicalDataReader(s.out_dir), Fixed({}), Fixed(_use_raw("absorbed"))
    )
    (r,) = [r for r in s.results if r.security == "AAA"]
    f = {b.day: float(b.f_close) for b in r.bars}
    for data in (loader.load("AAA"), loader.load("AAA", corrected=False)):
        assert data.corrections == ()
        assert all(b.adj_close == f[b.day] for b in data.history.bars)
        assert all(b.adj_volume is not None for b in data.history.bars)  # 读到 adj_volume
    view = loader.audit_view("AAA")
    assert view.corrections == () and view.corrected == view.raw
    assert all(b.adj_close == f[b.day] for b in view.corrected.bars)


def test_b15_legacy_reading_unchanged(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    lines = [",".join(BAR_COLUMNS[:-1])] + [f"{d},10,10,10,10,1,5,5,5,5" for d in DAYS[:6]]
    for sym in ("SPY", "AAA"):
        (raw / f"{sym}.csv").write_text("\n".join(lines) + "\n", "utf-8")
    loader = MarketDataLoader(CsvRawDataReader(raw), Fixed({}), Fixed(_use_raw(None)))
    bars = {b.day: b for b in loader.load("AAA").history.bars}
    assert bars[DAYS[3]].adj_close == 10.0 and bars[DAYS[0]].adj_close == 5.0  # use_raw 照旧作用
    assert bars[DAYS[3]].adj_volume is None


# ---- B16：原始价更正经配置文件 ----


def test_b16_raw_price_correction_via_config(tmp_path: Path) -> None:
    massive, cal, cfg = make_inputs(tmp_path)
    write_ticker(
        massive,
        "KKK",
        {
            DAYS[0]: "100",
            DAYS[1]: "50",
            DAYS[2]: ("49", "50", "48", "59"),  # 原始收盘有误（59，高于最高 50）
        },
        splits=[[DAYS[1].isoformat(), "1", "2"]],
    )
    with (cfg / "raw_price_corrections.csv").open("a", encoding="utf-8") as f:
        f.write(f"C1,KKK,{DAYS[2]},close,49,PM-T,测试,1\n")
    s = build(tmp_path, tmp_path / "out", symbols=["KKK"])
    lines = (s.out_dir / "KKK.csv").read_text("utf-8").splitlines()[1:]
    assert [ln.split(",")[9] for ln in lines] == ["100.000000", "100.000000", "98.000000"]
    assert lines[2].split(",")[4] == "49.000000"  # 原始列为更正后的原始价
    assert s.results[0].raw_correction_ids == ("C1",)


# ---- B21：版本身份 ----


class _Sink:
    def __init__(self) -> None:
        self.versions: list[str] = []

    def exists(self, version: str) -> bool:
        return False

    def write(self, output) -> Path:  # noqa: ANN001
        self.versions.append(output.version)
        return Path(output.version)


def test_b21_version_identity(tmp_path: Path) -> None:
    massive, cal, cfg = make_inputs(tmp_path)
    full = read_calendar(cal)
    # 日历比本证券的日线多出两个交易日（2015-05-01、05-04），它们与本证券无关
    calendar = replace(full, rows=(*full.rows, date(2015, 5, 1), date(2015, 5, 4)))
    config = CanonicalConfig(calendar, {}, (), None, (), None, None)
    write_status(massive)
    source = MassiveCanonicalSource(massive)

    def version(cfg_: CanonicalConfig, gaps: tuple[int, ...] = (10, 20)) -> str:
        sink = _Sink()
        build_canonical(
            ["AAA"],
            source,
            cfg_,
            sink,
            code_commit="c0ffee",
            params=default_params(gaps),
            clock=lambda: datetime(2026, 10, 18, tzinfo=UTC),
        )
        return sink.versions[0]

    v1 = version(config)
    assert version(config) == v1
    fewer = replace(calendar, rows=calendar.rows[:-1])  # 删去 2015-05-04
    assert version(replace(config, calendar=fewer)) != v1  # 只改交易日历
    assert version(config, (10,)) != v1  # 只改 gap_breaks


# ---- B22：构建 → 日志 → 区间审计 ----


def test_b22_build_log_then_interval_audit(built) -> None:  # noqa: ANN001
    _, s = built
    log = json.loads((s.out_dir / "AAA.adjlog.json").read_text("utf-8"))
    fields = {pair[0]: pair[1] for pair in log["map"]}
    actions = fields["actions"]["tuple"]
    groups = fields["groups"]["tuple"]
    div_actions = [a for a in actions if "dividends" in json.dumps(a, ensure_ascii=False)]
    div_groups = [g for g in groups if "|DIV|" in json.dumps(g, ensure_ascii=False)]
    assert len(div_actions) == 2 and len(div_groups) == 1
    (r,) = [r for r in s.results if r.security == "AAA"]
    (g,) = [g for g in r.groups if "|DIV|" in g.group_id]
    assert g.b == Decimal("0.97") and len(g.members) == 2
    with (s.out_dir / "AAA.csv").open(encoding="utf-8") as f:
        prices = [
            CanonicalBarPrices(
                date.fromisoformat(x["date"]), Decimal(x["close"]), Decimal(x["adj_close"])
            )
            for x in csv.DictReader(f)
        ]
    assert canonical_factor_checks(prices, r.groups) == []  # 预期因子 0.97，不报告跳变
    doubled = canonical_factor_checks(prices, (*r.groups, g))  # 若按原始行动相乘（0.97² = 0.9409）
    assert len(doubled) == 1 and doubled[0].expected_factor == Decimal("0.9409")


# ---- B24：规范模式的读取失败 ----


def _copy(s, tmp_path: Path, name: str) -> Path:  # noqa: ANN001
    dst = tmp_path / name / s.version
    shutil.copytree(s.out_dir, dst)
    return dst


def _reseal(root: Path) -> None:
    files = {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name not in ("MANIFEST.sha256", BUILD_META_FILE)
    }
    (root / "MANIFEST.sha256").write_bytes(manifest_text(files))


def _edit_eligibility(root: Path, fn) -> None:  # noqa: ANN001
    p = root / "_eligibility.csv"
    lines = p.read_text("utf-8").splitlines()
    p.write_text("\n".join(fn(lines)) + "\n", "utf-8")
    _reseal(root)


def test_b24_canonical_read_failures(built, tmp_path: Path) -> None:  # noqa: ANN001
    _, s = built
    ok = CanonicalDataReader(s.out_dir)
    assert ok.exists("AAA") and not ok.exists("BBB")  # diagnostic 下的不读
    with pytest.raises(CanonicalModeError, match="不可正式使用"):
        ok.load_text("BBB")
    # 缺行
    d = _copy(s, tmp_path, "a")
    _edit_eligibility(d, lambda ls: [ln for ln in ls if not ln.startswith("AAA,")])
    with pytest.raises(CanonicalModeError, match="恰好一行"):
        CanonicalDataReader(d).load_text("AAA")
    # 同一证券两行
    d = _copy(s, tmp_path, "b")
    _edit_eligibility(d, lambda ls: [*ls, next(ln for ln in ls if ln.startswith("AAA,"))])
    with pytest.raises(CanonicalModeError, match="恰好一行"):
        CanonicalDataReader(d).load("AAA")
    # 清单不符
    d = _copy(s, tmp_path, "c")
    (d / "AAA.csv").write_text((d / "AAA.csv").read_text("utf-8") + "\n", "utf-8")
    with pytest.raises(CanonicalModeError, match="哈希不符"):
        CanonicalDataReader(d)
    # 多出文件
    d = _copy(s, tmp_path, "d")
    (d / "diagnostic" / "ZZZ.csv").write_text("status\n", "utf-8")
    with pytest.raises(CanonicalModeError, match="多出"):
        CanonicalDataReader(d)
    # _dataset.json 缺字段
    d = _copy(s, tmp_path, "e")
    ds = json.loads((d / "_dataset.json").read_text("utf-8"))
    ds.pop("calendar")
    (d / "_dataset.json").write_text(json.dumps(ds), "utf-8")
    _reseal(d)
    with pytest.raises(CanonicalModeError, match="缺字段"):
        open_market_reader(d)
    # 版本号与目录名不符
    d = tmp_path / "f" / "d0a-000000000000"
    shutil.copytree(s.out_dir, d)
    with pytest.raises(CanonicalModeError, match="不一致"):
        open_market_reader(d)
    # 没有 _dataset.json 却有 _eligibility.csv：不按旧模式读
    d = _copy(s, tmp_path, "g")
    (d / "_dataset.json").unlink()
    with pytest.raises(CanonicalModeError, match="不按旧模式"):
        open_market_reader(d)
    with pytest.raises(CanonicalModeError):
        CsvRawDataReader(d)


# ---- B26：封存边界 ----


def test_b26_new_build_entry_counts_and_skips_sealed_rows(tmp_path: Path) -> None:
    massive, cal, cfg = make_inputs(tmp_path)
    sealed = "2020-01-02"
    for rel, line in (
        ("aggs/AAA_raw.csv", f"{sealed},非法,x,x,x,x,,,{HOST}"),
        ("actions/AAA_splits.csv", f"{sealed},非法,x,{HOST}"),
        ("actions/AAA_dividends.csv", f"{sealed},非法,CD,,,,,{HOST}"),
    ):
        with (massive / rel).open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    with cal.open("a", encoding="utf-8") as f:
        f.write(f"{sealed}\n")
    s = build(tmp_path, tmp_path / "out", symbols=["AAA"])
    assert s.sealed_rows_skipped == 4
    dataset = json.loads((s.out_dir / "_dataset.json").read_text("utf-8"))
    assert dataset["sealed_rows_skipped"] == 4
    assert dataset["calendar"]["sealed_rows_skipped"] == 1
    seg = dataset["securities"]["AAA"]["segments"][0]
    assert [seg[k]["sealed_rows_skipped"] for k in ("bars", "splits", "dividends")] == [1, 1, 1]


@pytest.mark.parametrize(
    ("file", "line"),
    [
        ("raw_price_corrections.csv", "C1,AAA,2020-01-02,close,非法,PM,证据,1"),
        (
            "adjustment_verifications.csv",
            "V1,group,AAA|DIV|2020-01-02|PER_SHARE|NONE,,ABNORMAL_CONFIRMED,非法,PM,证据",
        ),
        (
            "adjustment_verifications.csv",
            "V2,action,massive:actions/AAA_dividends.csv#1,abc,DECL_RECORD_CORRECTED,2020-01-02,PM,证据",
        ),
    ],
)
def test_b26_manual_config_sealed_dates_raise_seal_error(
    tmp_path: Path, file: str, line: str
) -> None:
    _, _, cfg = make_inputs(tmp_path)
    with (cfg / file).open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    with pytest.raises(SealViolation):  # 报封存越界，不报数值错误
        build(tmp_path, tmp_path / "out", symbols=["AAA"])
    assert not (tmp_path / "out").exists()


def test_dataset_records_config_hashes(built) -> None:  # noqa: ANN001
    base, s = built
    dataset = json.loads((s.out_dir / "_dataset.json").read_text("utf-8"))
    cfg = base / "config"
    sha = hashlib.sha256((cfg / "price_corrections.csv").read_bytes()).hexdigest()
    assert dataset["config"]["price_corrections_sha256"] == sha
    assert dataset["calendar"]["first"] == DAYS[0].isoformat()
    assert dataset["calendar"]["last"] == DAYS[-1].isoformat()
