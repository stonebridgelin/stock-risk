import json
from datetime import date
from pathlib import Path

import pytest

from stock_risk.application.fetch_raw_data import FetchRawDataService
from stock_risk.domain.market_data import DailyBar, PriceHistory
from stock_risk.domain.segments import SymbolSegment
from stock_risk.domain.universe import Role, UniverseMember
from stock_risk.infrastructure.storage.csv_raw_repository import CsvRawDataRepository


def bar(d: date) -> DailyBar:
    return DailyBar(d, 10, 11, 9, 10.5, 1000, 10, 11, 9, 10.5)


class RangeSource:
    """按请求区间返回日线；记录每次请求的代码与区间。"""

    name = "Fake"

    def __init__(self, days: dict[str, list[date]]) -> None:
        self.days = days
        self.calls: list[tuple[str, date, date]] = []

    def fetch(self, ticker: str, start: date, end: date) -> PriceHistory | None:
        self.calls.append((ticker, start, end))
        if ticker not in self.days:
            return None
        return PriceHistory(tuple(bar(d) for d in self.days[ticker] if start <= d <= end), ())


class Fixed:
    def __init__(self, value) -> None:
        self.value = value

    def load(self):
        return self.value


VRX_DAYS = [date(2018, 7, 12), date(2018, 7, 13)]
BHC_DAYS = [date(2018, 7, 16), date(2018, 7, 17), date(2020, 1, 2)]
SEGS = {
    "VRX": (
        SymbolSegment("VRX", "VRX", None, date(2018, 7, 13)),
        SymbolSegment("VRX", "BHC", date(2018, 7, 16), None),
    )
}
MEMBERS = [
    UniverseMember("VRX", ("VRX", "BHC"), "event", Role.SIGNAL_TARGET),
    UniverseMember("AAPL", ("AAPL",), "x", Role.SIGNAL_TARGET),
]


def make(tmp_path: Path, days: dict[str, list[date]], segs=SEGS):
    src = RangeSource(days)
    raw = tmp_path / "raw"
    svc = FetchRawDataService(
        src, Fixed(MEMBERS), CsvRawDataRepository(raw), lambda _: None, segments=Fixed(segs)
    )
    return svc, src, raw


def test_segmented_fetch_stitches_and_records_segments(tmp_path: Path) -> None:
    svc, src, raw = make(tmp_path, {"VRX": VRX_DAYS, "BHC": BHC_DAYS})
    report = svc.run(date(2008, 1, 1), symbols=["vrx"])

    (o,) = report.outcomes
    assert o.ok and o.source_ticker == "VRX+BHC" and o.rows == 4
    assert o.last_day == date(2018, 7, 17)
    assert src.calls == [
        ("VRX", date(2008, 1, 1), date(2018, 7, 13)),
        ("BHC", date(2018, 7, 16), date(2019, 12, 31)),
    ]
    lines = (raw / "VRX.csv").read_text("utf-8").splitlines()
    assert [ln[:10] for ln in lines[1:]] == ["2018-07-12", "2018-07-13", "2018-07-16", "2018-07-17"]
    status = json.loads((raw / "_fetch_status.json").read_text("utf-8"))
    assert [s["source_ticker"] for s in status["VRX"]["segments"]] == ["VRX", "BHC"]
    assert "分段拼接: VRX 2018-07-12 ~ 2018-07-13（2 行），BHC" in (raw / "README.txt").read_text(
        "utf-8"
    )


def test_segment_without_data_fails_whole_symbol(tmp_path: Path) -> None:
    svc, _, raw = make(tmp_path, {"VRX": VRX_DAYS})
    (o,) = svc.run(date(2008, 1, 1), symbols=["VRX"]).outcomes
    assert not o.ok and "BHC" in (o.error or "")
    assert not (raw / "VRX.csv").exists()


def test_symbols_filter_rejects_unknown(tmp_path: Path) -> None:
    svc, _, _ = make(tmp_path, {})
    with pytest.raises(ValueError):
        svc.run(date(2008, 1, 1), symbols=["ZZZ"])


def test_segment_config_for_unknown_symbol_rejected(tmp_path: Path) -> None:
    segs = {"ZZZ": (SymbolSegment("ZZZ", "ZZZ"),)}
    svc, _, _ = make(tmp_path, {}, segs)
    with pytest.raises(ValueError):
        svc.run(date(2008, 1, 1))


def test_unsegmented_symbol_keeps_candidate_behavior(tmp_path: Path) -> None:
    svc, src, _ = make(tmp_path, {"AAPL": [date(2019, 1, 2)]})
    (o,) = svc.run(date(2008, 1, 1), symbols=["AAPL"]).outcomes
    assert o.ok and o.source_ticker == "AAPL" and o.segments == ()
    assert src.calls == [("AAPL", date(2008, 1, 1), date(2019, 12, 31))]
