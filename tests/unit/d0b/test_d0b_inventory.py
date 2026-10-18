"""T12-D0b 第 2.1、2.7 节：证券身份认定与映射比较。

补取命令的确认与 --dry-run（不读设置、不联网）。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import stock_risk.bootstrap as bootstrap
from stock_risk.application.d0b_inventory import (
    IdentityQuestion,
    VerifyEvidence,
    compare_mapping,
    determine_identity,
    effective_range,
    gap_specs,
    resolve_segments,
)
from stock_risk.config.settings import Secret
from stock_risk.domain.listing import ListingPeriod
from stock_risk.domain.segments import SymbolSegment
from stock_risk.domain.universe import Role, UniverseMember
from stock_risk.infrastructure.storage.d0b_files import write_gaps, write_identity
from stock_risk.interfaces.cli.main import main
from tests.unit.d0b.fakes import FakeMassive, bar_json, ok, weekdays

CAL = tuple(weekdays(date(2008, 1, 2), date(2019, 12, 31)))
E = date(2019, 12, 31)


def member(sym: str, cands: str | None = None) -> UniverseMember:
    return UniverseMember(sym, tuple((cands or sym).split("|")), "test", Role.SIGNAL_TARGET)


def ev(sym: str, rows: int, last: date | None, tiingo_last: date | None = E) -> VerifyEvidence:
    return VerifyEvidence(
        f"verify/prices_{sym}.csv",
        rows,
        date(2010, 1, 4) if rows else None,
        last,
        2516,
        tiingo_last,
    )


def test_identity_rules() -> None:
    listing = {
        "WFX": ListingPeriod("WFX", None, date(2019, 5, 10), "NYSE", "https://x.invalid", None, "")
    }
    rows = determine_identity(
        [member("AAA"), member("VRY", "VRY|BHY"), member("WFX"), member("SNY", "SNY|SNZ")],
        {
            "AAA": ev("AAA", 2516, E),
            "VRY": ev("VRY", 2147, date(2018, 7, 13)),
            "WFX": ev("WFX", 2354, date(2019, 5, 10)),
            "SNY": ev("SNY", 0, None),
        },
        listing,
        {"SNY": ((SymbolSegment("SNY", "SNZ"),), "PM-T")},
        CAL,
    )
    seg = {r.symbol: [s.text() for s in r.segments] for r in rows}
    assert seg["AAA"] == ["AAA:2008-01-01~2019-12-31"]
    assert seg["VRY"] == ["VRY:2008-01-01~2018-07-13", "BHY:2018-07-16~2019-12-31"]
    assert seg["WFX"] == ["WFX:2008-01-01~2019-05-10"]
    assert seg["SNY"] == ["SNZ:2008-01-01~2019-12-31"]
    assert "PM-T" in rows[3].evidence
    assert len(gap_specs(rows)) == (1 + 2 + 1 + 1) * 4


def test_identity_questions_stop() -> None:
    with pytest.raises(IdentityQuestion, match="需要裁决"):
        determine_identity([member("SNY", "SNY|SNZ")], {"SNY": ev("SNY", 0, None)}, {}, {}, CAL)
    with pytest.raises(IdentityQuestion, match="没有下一个候选"):
        determine_identity([member("ZZZ")], {"ZZZ": ev("ZZZ", 100, date(2015, 1, 2))}, {}, {}, CAL)
    with pytest.raises(IdentityQuestion, match="没有中转服务核验文件"):
        determine_identity([member("QQQ")], {}, {}, {}, CAL)


def test_resolve_segments_precedence_and_compare() -> None:
    sym_segs = {
        "VRY": (
            SymbolSegment("VRY", "VRY", None, date(2018, 7, 13)),
            SymbolSegment("VRY", "BHY", date(2018, 7, 16), None),
        )
    }
    massive = {"SNY": (SymbolSegment("SNY", "SNZ"),)}
    assert (
        resolve_segments("SNY", {"SNY": (SymbolSegment("SNY", "XXX"),)}, massive)[0].source_ticker
        == "SNZ"
    )
    assert [s.source_ticker for s in resolve_segments("VRY", sym_segs, massive)] == ["VRY", "BHY"]
    assert resolve_segments("AAA", sym_segs, massive)[0].source_ticker == "AAA"
    rows = determine_identity(
        [member("SNY", "SNY|SNZ")],
        {"SNY": ev("SNY", 0, None)},
        {},
        {"SNY": (massive["SNY"], "PM-T")},
        CAL,
    )
    assert all(c.same for c in compare_mapping(rows, sym_segs, massive, {}))
    (diff,) = compare_mapping(rows, sym_segs, {}, {})  # 构建器不读源代码映射时：不一致
    assert not diff.same and diff.builder == "SNY:2008-01-01~2019-12-31"
    assert effective_range(None) == (date(2008, 1, 1), E)


# ---- 命令行：--dry-run 与确认 ----


def _files(tmp_path: Path) -> tuple[Path, Path]:
    rows = determine_identity([member("AAA")], {"AAA": ev("AAA", 2516, E)}, {}, {}, CAL)
    identity = write_identity(tmp_path / "identity.csv", rows)
    gaps = write_gaps(
        tmp_path / "gaps.csv",
        gap_specs(rows),
        {c: "测试" for c in ("raw", "adj", "splits", "dividends")},
    )
    return identity, gaps


def _no_settings() -> None:
    raise AssertionError("不得读取设置（.env）")


def test_cli_dry_run_reads_no_settings_and_sends_nothing(
    tmp_path: Path, monkeypatch, capsys
) -> None:  # noqa: ANN001
    identity, gaps = _files(tmp_path)
    monkeypatch.setattr(bootstrap, "load_settings", _no_settings)
    out = tmp_path / "download"
    code = main(
        [
            "fetch-massive-d0b",
            "--identity",
            str(identity),
            "--gaps",
            str(gaps),
            "--out",
            str(out),
            "--dry-run",
        ]
    )
    text = capsys.readouterr().out
    assert code == 0 and "未联网" in text and "预计请求总次数：4" in text
    assert not out.exists()


def test_cli_requires_yes(tmp_path: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    identity, gaps = _files(tmp_path)
    monkeypatch.setattr(bootstrap, "load_settings", _no_settings)
    monkeypatch.setattr("builtins.input", lambda _: "no")
    out = tmp_path / "download"
    code = main(
        ["fetch-massive-d0b", "--identity", str(identity), "--gaps", str(gaps), "--out", str(out)]
    )
    assert code == 1 and "未确认" in capsys.readouterr().out
    assert not out.exists()


def test_cli_yes_runs_through_fake_transport(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    identity, gaps = _files(tmp_path)
    fake = FakeMassive({("raw", "AAA"): ok([bar_json(date(2015, 3, 2), "1", "1", "1", "1")])})

    class S:
        massive_advanced_base_url = "https://relaynex.vip"
        massive_advanced_api_key = Secret("test-key")
        massive_per_minute = 600

    monkeypatch.setattr(bootstrap, "load_settings", lambda: S())
    monkeypatch.setattr(bootstrap, "make_transport", lambda parse: fake.transport)
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    import stock_risk.infrastructure.massive.client as client

    monkeypatch.setattr(client.time, "sleep", lambda _: None)
    out = tmp_path / "download"
    code = main(
        ["fetch-massive-d0b", "--identity", str(identity), "--gaps", str(gaps), "--out", str(out)]
    )
    assert code == 0 and len(fake.calls) == 4
    assert (out / "aggs/AAA_raw.csv").exists() and (out / "_ledger.json").exists()


def test_cli_refuses_non_relay_host(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    identity, gaps = _files(tmp_path)

    class S:
        massive_advanced_base_url = "https://api.massive.com"
        massive_advanced_api_key = Secret("test-key")
        massive_per_minute = 600

    monkeypatch.setattr(bootstrap, "load_settings", lambda: S())
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    with pytest.raises(SystemExit, match="只经中转服务"):
        main(
            [
                "fetch-massive-d0b",
                "--identity",
                str(identity),
                "--gaps",
                str(gaps),
                "--out",
                str(tmp_path / "d"),
            ]
        )


def test_gap_not_in_identity_refused(tmp_path: Path) -> None:
    identity, gaps = _files(tmp_path)
    gaps.write_text(
        gaps.read_text("utf-8").replace("AAA,AAA,raw,2008-01-01", "AAA,AAB,raw,2008-01-01"), "utf-8"
    )
    with pytest.raises(SystemExit, match="不符"):
        main(
            [
                "fetch-massive-d0b",
                "--identity",
                str(identity),
                "--gaps",
                str(gaps),
                "--out",
                str(tmp_path / "d"),
                "--dry-run",
            ]
        )
