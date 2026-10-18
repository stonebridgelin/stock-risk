"""B8：adj_volume 的完整读写链（T11-D0a 第 2 节）。

五种输入（缺列、空单元格、零、正数、非法值）× 三个入口（parse_bar、load、load_text）；写出往返；
Tiingo 映射不给 adj_volume；扫描 src 中读取 adj_volume 的地方，没有回退到 volume 的写法。
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from stock_risk.application.models import FetchOutcome
from stock_risk.domain.market_data import DailyBar, PriceHistory
from stock_risk.domain.numeric import InvalidInputError
from stock_risk.domain.raw_text import RawBarText, parse_bar
from stock_risk.domain.universe import Role, UniverseMember
from stock_risk.infrastructure.storage.csv_raw_reader import CsvRawDataReader
from stock_risk.infrastructure.storage.csv_raw_repository import BAR_COLUMNS, CsvRawDataRepository
from stock_risk.infrastructure.tiingo.mapper import to_history

D1 = date(2015, 3, 2)
OLD_HEADER = "date,open,high,low,close,volume,adj_open,adj_high,adj_low,adj_close"
SRC = Path(__file__).resolve().parents[3] / "src"


def text_row(adj_volume: str | None) -> RawBarText:
    return RawBarText(D1, "10", "11", "9", "10", "1000", "10", "11", "9", "10", adj_volume)


# ---- parse_bar ----


@pytest.mark.parametrize(
    ("text", "expected"),
    [(None, None), ("", None), ("0", 0.0), ("0.0", 0.0), ("1100", 1100.0)],
)
def test_parse_bar_reads(text: str | None, expected: float | None) -> None:
    assert parse_bar("X", text_row(text)).adj_volume == expected


@pytest.mark.parametrize("text", ["-1", "nan", "inf", "abc"])
def test_parse_bar_rejects_illegal(text: str) -> None:
    with pytest.raises(InvalidInputError):
        parse_bar("X", text_row(text))


# ---- csv_raw_reader.load / load_text ----


def _write(root: Path, header: str, cells: list[str]) -> CsvRawDataReader:
    root.mkdir(parents=True, exist_ok=True)
    lines = [header, *(f"{D1},10,11,9,10,1000,10,11,9,10{c}" for c in cells)]
    (root / "X.csv").write_text("\n".join(lines) + "\n", "utf-8")
    return CsvRawDataReader(root)


def test_reader_old_file_without_column(tmp_path: Path) -> None:
    reader = _write(tmp_path, OLD_HEADER, [""])
    assert reader.load("X").history.bars[0].adj_volume is None
    (row,) = reader.load_text("X").bars
    assert row.adj_volume is None and parse_bar("X", row).adj_volume is None


@pytest.mark.parametrize(
    ("cell", "text", "value"),
    [("", None, None), ("0", "0", 0.0), ("0.0", "0.0", 0.0), ("1100", "1100", 1100.0)],
)
def test_reader_with_column(
    tmp_path: Path, cell: str, text: str | None, value: float | None
) -> None:
    reader = _write(tmp_path, OLD_HEADER + ",adj_volume", [f",{cell}"])
    assert reader.load("X").history.bars[0].adj_volume == value
    (row,) = reader.load_text("X").bars
    assert row.adj_volume == text  # load_text 只取文本：空单元格为 None，其余为原文
    assert parse_bar("X", row).adj_volume == value


@pytest.mark.parametrize("cell", ["-1", "nan", "inf", "abc"])
def test_reader_rejects_illegal(tmp_path: Path, cell: str) -> None:
    reader = _write(tmp_path, OLD_HEADER + ",adj_volume", [f",{cell}"])
    with pytest.raises(InvalidInputError):
        reader.load("X")
    (row,) = reader.load_text("X").bars
    assert row.adj_volume == cell  # 原文照取，由解析报错
    with pytest.raises(InvalidInputError):
        parse_bar("X", row)


# ---- 写出往返 ----


def test_write_round_trip_keeps_none_zero_positive(tmp_path: Path) -> None:
    assert BAR_COLUMNS[-1] == "adj_volume"
    days = [date(2015, 3, 2), date(2015, 3, 3), date(2015, 3, 4)]
    bars = tuple(
        DailyBar(d, 10, 11, 9, 10, 1000, 10, 11, 9, 10, v)
        for d, v in zip(days, (None, 0.0, 1100.0), strict=True)
    )
    repo = CsvRawDataRepository(tmp_path)
    member = UniverseMember("X", ("X",), "test", Role.SIGNAL_TARGET)
    repo.save(member, PriceHistory(bars, ()), FetchOutcome("X", "test", True))
    lines = (tmp_path / "X.csv").read_text("utf-8").splitlines()
    assert lines[0].endswith(",adj_volume")
    assert [ln.rsplit(",", 1)[1] for ln in lines[1:]] == ["", "0.0", "1100.0"]
    reader = CsvRawDataReader(tmp_path)
    assert [b.adj_volume for b in reader.load("X").history.bars] == [None, 0.0, 1100.0]
    parsed = [parse_bar("X", r).adj_volume for r in reader.load_text("X").bars]
    assert parsed == [None, 0.0, 1100.0]


def test_tiingo_mapper_leaves_adj_volume_unknown() -> None:
    h = to_history(
        [
            {
                "date": "2015-03-02T00:00:00.000Z",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 1000,
                "adjOpen": 5,
                "adjHigh": 5.5,
                "adjLow": 4.5,
                "adjClose": 5.25,
                "adjVolume": 2000,
                "divCash": 0.0,
                "splitFactor": 1.0,
            }
        ]
    )
    assert h.bars[0].adj_volume is None  # 不用 volume 或供应商的调整量填充


# ---- 回退扫描（第 2.4 节） ----

FALLBACK = re.compile(r"adj_volume\b[^#\n]*\b(or|else)\b[^#\n]*\bvolume\b")
FALLBACK_REVERSE = re.compile(r"\bvolume\b[^#\n]*\bif\b[^#\n]*adj_volume\s+is\s+None")


def test_no_fallback_from_adj_volume_to_volume() -> None:
    reads: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        for no, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
            if "adj_volume" not in line:
                continue
            reads.append(f"{path.relative_to(SRC)}:{no}")
            code = line.split("#", 1)[0]
            assert not FALLBACK.search(code), f"{path.name}:{no} 回退到 volume：{line.strip()}"
            assert not FALLBACK_REVERSE.search(code), f"{path.name}:{no}：{line.strip()}"
    assert reads  # 扫描确实覆盖到了读取点
