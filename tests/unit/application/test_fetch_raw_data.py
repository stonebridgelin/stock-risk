from datetime import date
from pathlib import Path

import pytest

from stock_risk.application.fetch_raw_data import FetchRawDataService
from stock_risk.application.ports import PriceSourceError
from stock_risk.domain.market_data import DailyBar, PriceHistory
from stock_risk.domain.universe import UniverseMember
from stock_risk.infrastructure.storage.csv_raw_repository import CsvRawDataRepository


def bar(d: date) -> DailyBar:
    return DailyBar(d, 10, 11, 9, 10.5, 1000, 5, 5.5, 4.5, 5.25)


class FakeSource:
    name = "Fake"

    def __init__(self, data: dict[str, PriceHistory | None | Exception]) -> None:
        self.data = data
        self.calls: list[str] = []

    def fetch(self, ticker: str, start: date, end: date) -> PriceHistory | None:
        self.calls.append(ticker)
        v = self.data.get(ticker)
        if isinstance(v, Exception):
            raise v
        return v


class FakeUniverse:
    def __init__(self, members: list[UniverseMember]) -> None:
        self.members = members

    def load(self) -> list[UniverseMember]:
        return self.members


UNIVERSE = [
    UniverseMember("AAPL", ("AAPL",), "large_drawdown"),
    UniverseMember("FB", ("FB", "META"), "ipo"),
    UniverseMember("SHLD", ("SHLD",), "delisted"),
    UniverseMember("BAD", ("BAD",), "x"),
]


def make(tmp_path: Path, data: dict) -> tuple[FetchRawDataService, FakeSource, Path]:
    src = FakeSource(data)
    raw = tmp_path / "raw"
    svc = FetchRawDataService(
        src, FakeUniverse(UNIVERSE), CsvRawDataRepository(raw), lambda _: None
    )
    return svc, src, raw


DATA = {
    "AAPL": PriceHistory(
        (bar(date(2019, 12, 30)), bar(date(2019, 12, 31)), bar(date(2020, 1, 2))), ()
    ),
    "FB": None,
    "META": PriceHistory((bar(date(2012, 5, 18)),), ()),
    "SHLD": None,
    "BAD": PriceSourceError("HTTP 500"),
}


def test_run_saves_truncates_and_falls_back(tmp_path: Path) -> None:
    svc, src, raw = make(tmp_path, DATA)
    report = svc.run(date(2008, 1, 1))

    by = {o.symbol: o for o in report.outcomes}
    assert by["AAPL"].ok and by["AAPL"].last_day == date(2019, 12, 31) and by["AAPL"].rows == 2
    assert by["FB"].ok and by["FB"].source_ticker == "META"
    assert not by["SHLD"].ok and not by["BAD"].ok
    assert src.calls == ["AAPL", "FB", "META", "SHLD", "BAD"]

    aapl = (raw / "AAPL.csv").read_text("utf-8").splitlines()
    assert aapl[0].startswith("date,open,high,low,close,volume,adj_open")
    assert aapl[-1].startswith("2019-12-31")
    assert not any(line.startswith("2020") for line in aapl)
    assert (raw / "FB.csv").exists()
    readme = (raw / "README.txt").read_text("utf-8")
    assert "成功 2 个" in readme and "缺失 2 个" in readme and "数据源代码 META" in readme
    assert len((raw / "MANIFEST.sha256").read_text("utf-8").split()) == 4


def test_run_resumes_and_skips_fetched(tmp_path: Path) -> None:
    svc, _, raw = make(tmp_path, DATA)
    svc.run(date(2008, 1, 1))
    svc2 = FetchRawDataService(
        FakeSource(DATA), FakeUniverse(UNIVERSE), CsvRawDataRepository(raw), lambda _: None
    )
    report = svc2.run(date(2008, 1, 1))
    skipped = {o.symbol for o in report.outcomes if o.skipped}
    assert skipped == {"AAPL", "FB"}
    assert "成功 2 个" in (raw / "README.txt").read_text("utf-8")


def test_run_rejects_end_after_seal(tmp_path: Path) -> None:
    svc, _, _ = make(tmp_path, DATA)
    with pytest.raises(ValueError):
        svc.run(date(2008, 1, 1), date(2020, 1, 1))
