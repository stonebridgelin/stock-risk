from datetime import date
from pathlib import Path

from stock_risk.application.audit_raw_data import AuditRawDataService
from stock_risk.application.market_data_loader import MarketDataLoader
from stock_risk.domain.corrections import CorrectionMethod, PriceCorrection
from stock_risk.domain.listing import ListingPeriod
from stock_risk.domain.market_data import DailyBar, PriceHistory
from stock_risk.domain.segments import SymbolSegment
from stock_risk.domain.universe import Role, UniverseMember
from stock_risk.infrastructure.storage.audit_report_writer import AuditReportFileWriter
from stock_risk.infrastructure.storage.csv_raw_reader import CsvRawDataReader
from stock_risk.infrastructure.storage.csv_raw_repository import BAR_COLUMNS

CAL = [date(2018, 7, d) for d in (10, 11, 12, 13, 16, 17)]


def write_raw(root: Path, symbol: str, rows: list[str], actions: list[str] = ()) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{symbol}.csv").write_text("\n".join([",".join(BAR_COLUMNS), *rows]) + "\n", "utf-8")
    (root / f"{symbol}.actions.csv").write_text(
        "\n".join(["date,type,value", *actions]) + "\n", "utf-8"
    )


def row(d: date, c: float = 10, adj_c: float | None = None, v: float = 100) -> str:
    a = c if adj_c is None else adj_c
    return f"{d},{c},{c + 1},{c - 1},{c},{v},{a},{a + 1},{a - 1},{a}"


class Fixed:
    def __init__(self, value) -> None:
        self.value = value

    def load(self):
        return self.value


def test_audit_end_to_end(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_raw(raw, "SPY", [row(d) for d in CAL])
    # VRX：缺 07-11；07-16 拼接；07-17 系数跳变且无公司行动；07-12 零成交量
    write_raw(
        raw,
        "VRX",
        [
            row(CAL[0]),
            row(CAL[2], v=0),
            row(CAL[3], c=20),
            row(CAL[4], c=21),
            row(CAL[5], c=21, adj_c=18),
        ],
    )
    # SHLD：退市日 07-12 之后的场外行不计缺口；07-11 收盘高于最高价
    write_raw(
        raw, "SHLD", [row(CAL[0]), f"{CAL[1]},10,11,9,12,100,10,11,9,10", row(CAL[2]), row(CAL[4])]
    )
    (raw / "MANIFEST.sha256").write_text("abc  SPY.csv\n", "utf-8")
    members = [
        UniverseMember(s, (s,), "x", Role.SIGNAL_TARGET) for s in ("SPY", "VRX", "SHLD", "GONE")
    ]
    segs = {
        "VRX": (
            SymbolSegment("VRX", "VRX", None, CAL[3]),
            SymbolSegment("VRX", "BHC", CAL[4], None),
        )
    }
    listings = {"SHLD": ListingPeriod("SHLD", listed_until=CAL[2], source_url="https://x")}

    corrections = {
        "VRX": (PriceCorrection("VRX", CAL[5], CAL[5], CorrectionMethod.USE_RAW, "PM-T", "测试"),)
    }
    loader = MarketDataLoader(CsvRawDataReader(raw), Fixed(listings), Fixed(corrections))
    svc = AuditRawDataService(loader, Fixed(members), Fixed(segs), AuditReportFileWriter())
    report, paths = svc.run(date(2026, 10, 13), tmp_path / "audit")

    by = {s.symbol: s for s in report.symbols}
    assert report.missing == ["GONE"] and report.manifest_sha256
    assert by["VRX"].gaps == (CAL[1],)
    assert by["VRX"].zero_volume == (CAL[2],)
    assert [j.day for j in by["VRX"].factor_jumps] == [CAL[5]]
    (j,) = by["VRX"].junctions
    assert j.open_to_close_ratio == 21 / 20
    assert by["SHLD"].gaps == () and by["SHLD"].rows_outside_listing == 1
    assert [f.kind.value for f in by["SHLD"].failures] == ["close_outside_range"]
    assert report.total_sealed_rows == 0
    # 更正：原始数据仍报告跳变，更正后不再出现
    assert by["VRX"].corrections == corrections["VRX"]
    assert by["VRX"].corrected_factor_jumps == ()
    assert by["SHLD"].corrections == () and by["SHLD"].corrected_factor_jumps == ()
    # MANIFEST：SPY 哈希不符，VRX、SHLD 不在清单中
    statuses = {m.file: m.status.value for m in report.manifest_mismatches}
    assert statuses == {
        "SPY.csv": "hash_mismatch",
        "VRX.csv": "not_in_manifest",
        "SHLD.csv": "not_in_manifest",
    }
    # D-02：明确失败、哈希不符、缺少原始文件使 audit-raw 返回非零
    # B-02（Gemini 第二阶段）：被审计证券的 CSV 不在清单中同样阻断
    assert report.blocking_problems == [
        "明确失败 1 项",
        "MANIFEST 哈希不符 1 个：SPY.csv",
        "被审计证券不在 MANIFEST 中 2 个：SHLD.csv, VRX.csv",
        "缺少原始文件 1 个：GONE",
    ]

    out = tmp_path / "audit" / "2026-10-13"
    assert {p.name for p in paths} >= {"summary.md", "explicit_failures.csv", "junctions.csv"}
    summary = (out / "summary.md").read_text("utf-8")
    assert "VRX → BHC" in summary and "缺少原始文件 1 个：GONE" in summary
    assert "## 已登记的更正" in summary and "PM-T" in summary and "- VRX：无" in summary
    assert "## MANIFEST 校验" in summary and "hash_mismatch" in summary
    assert (out / "manifest_check.csv").exists() and (out / "price_corrections.csv").exists()


def test_reader_counts_sealed_rows_without_returning_them(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_raw(
        raw, "X", [row(date(2019, 12, 31)), row(date(2020, 1, 2))], ["2020-01-02,dividend,0.1"]
    )
    f = CsvRawDataReader(raw).load("X")
    assert f.history.last_day == date(2019, 12, 31) and f.history.actions == ()
    assert f.sealed_rows == 2


def test_reader_rejects_unsorted_dates(tmp_path: Path) -> None:
    import pytest

    raw = tmp_path / "raw"
    write_raw(raw, "X", [row(date(2019, 1, 3)), row(date(2019, 1, 2))])
    with pytest.raises(ValueError):
        CsvRawDataReader(raw).load("X")


def test_history_type_is_price_history(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_raw(raw, "X", [row(date(2019, 1, 2))])
    assert isinstance(CsvRawDataReader(raw).load("X").history, PriceHistory)
    assert isinstance(CsvRawDataReader(raw).load("X").history.bars[0], DailyBar)


def _report(manifest_sha256, entries, mismatches=(), symbols=("X",)):
    from stock_risk.application.audit_models import AuditReport, SymbolAudit

    audits = [
        SymbolAudit(s, 1, None, None, 0, None, None, None, 0, (), (), (), (), ()) for s in symbols
    ]
    return AuditReport(
        date(2026, 10, 13),
        manifest_sha256,
        "SPY",
        None,
        None,
        0.1,
        symbols=audits,
        manifest_entries=entries,
        manifest_mismatches=list(mismatches),
    )


def test_blocking_problems_empty_when_clean() -> None:
    assert _report("abc", 1).blocking_problems == []


def test_b02_manifest_missing_or_empty_blocks() -> None:
    assert _report(None, 0).blocking_problems == ["MANIFEST.sha256 缺失"]
    assert _report("e3b0", 0).blocking_problems == ["MANIFEST.sha256 为空"]


def test_b02_audited_csv_not_in_manifest_blocks_others_only_reported() -> None:
    from stock_risk.domain.audit import ManifestMismatch, ManifestStatus

    nim = ManifestStatus.NOT_IN_MANIFEST
    mismatches = [ManifestMismatch(f, nim, None, "h") for f in ("X.csv", "Y.csv")]
    assert _report("abc", 1, mismatches).blocking_problems == [
        "被审计证券不在 MANIFEST 中 1 个：X.csv"
    ]


def test_b02_missing_manifest_file_end_to_end(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_raw(raw, "SPY", [row(d) for d in CAL])
    members = [UniverseMember("SPY", ("SPY",), "x", Role.SIGNAL_TARGET)]
    loader = MarketDataLoader(CsvRawDataReader(raw), Fixed({}), Fixed({}))
    svc = AuditRawDataService(loader, Fixed(members), Fixed({}), AuditReportFileWriter())
    report, _ = svc.run(date(2026, 10, 13), tmp_path / "audit")
    assert report.blocking_problems == [
        "MANIFEST.sha256 缺失",
        "被审计证券不在 MANIFEST 中 1 个：SPY.csv",
    ]
