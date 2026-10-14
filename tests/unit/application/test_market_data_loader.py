"""统一取数入口（I12）：封存检查 → 解析与量化 → 更正 → 观察区间截取 → 第二次封存检查。"""

from datetime import date, timedelta
from pathlib import Path

import pytest

from stock_risk.application.check_calendar import CheckCalendarService, compare_calendars
from stock_risk.application.market_data_loader import MarketDataLoader
from stock_risk.domain.corrections import CorrectionMethod, PriceCorrection
from stock_risk.domain.listing import ListingPeriod
from stock_risk.domain.numeric import InvalidInputError
from stock_risk.domain.observation import LeftEnd, RightEnd
from stock_risk.domain.seal import SealViolation
from stock_risk.infrastructure.storage.csv_raw_reader import CsvRawDataReader
from stock_risk.infrastructure.storage.csv_raw_repository import BAR_COLUMNS


class Fixed:
    def __init__(self, value) -> None:
        self.value = value

    def load(self):
        return self.value


def weekdays(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def row(d: date, c: str = "10", adj_c: str | None = None) -> str:
    a = c if adj_c is None else adj_c
    return f"{d},{c},{c},{c},{c},100,{a},{a},{a},{a}"


def write(root: Path, symbol: str, rows: list[str], actions: list[str] = ()) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{symbol}.csv").write_text("\n".join([",".join(BAR_COLUMNS), *rows]) + "\n", "utf-8")
    (root / f"{symbol}.actions.csv").write_text(
        "\n".join(["date,type,value", *actions]) + "\n", "utf-8"
    )


SPY_DAYS = weekdays(date(2019, 12, 2), date(2019, 12, 31))


def loader(raw: Path, listings=None, corrections=None) -> MarketDataLoader:
    if not (raw / "SPY.csv").exists():
        write(raw, "SPY", [row(d) for d in SPY_DAYS])
    return MarketDataLoader(CsvRawDataReader(raw), Fixed(listings or {}), Fixed(corrections or {}))


def test_sealed_row_raises_before_any_price_is_parsed(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    # 2020-01-02 行的价格是无法解析的文本：若先解析就会抛 InvalidInputError 而不是 SealViolation
    write(raw, "X", [row(date(2019, 12, 31)), "2020-01-02,abc,abc,abc,abc,x,abc,abc,abc,abc"])
    with pytest.raises(SealViolation):
        loader(raw).load("X")
    # 审计路径对同一文件只计数、不报错
    view = loader(raw).audit_view("X")
    assert view.sealed_rows == 1 and view.raw.last_day == date(2019, 12, 31)


def test_sealed_action_row_also_raises(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write(raw, "X", [row(date(2019, 12, 31))], ["2020-01-02,dividend,0.1"])
    with pytest.raises(SealViolation):
        loader(raw).load("X")


def test_prices_quantized_from_text(tmp_path: Path) -> None:
    # 13 号 S21h 的读取部分：复权收盘文本 "96.0000004" → 96.000000
    raw = tmp_path / "raw"
    write(raw, "X", [row(date(2019, 12, 2), c="96.0000004")])
    bar = loader(raw).load("X").history.bars[0]
    assert bar.adj_close == 96.0 and bar.close == 96.0


def test_explicit_failure_raises(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write(raw, "X", [f"{date(2019, 12, 2)},10,9,11,10,100,10,11,9,10"])  # 原始价 高 < 低
    with pytest.raises(InvalidInputError):
        loader(raw).load("X")


def test_tail_missing_and_ranges(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    days = weekdays(date(2019, 12, 2), date(2019, 12, 20))
    write(raw, "X", [row(d) for d in days])
    data = loader(raw).load("X")
    assert data.missing_days == tuple(weekdays(date(2019, 12, 23), date(2019, 12, 31)))
    assert data.ranges.right_end is RightEnd.EXPLORATION_END
    # 数据首日 12-02 等于本测试日历中下载起点后的首个交易日
    assert data.ranges.left_end is LeftEnd.REQUEST_START


def test_corrections_then_clip_to_observation(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    days = weekdays(date(2019, 12, 2), date(2019, 12, 13))
    write(raw, "X", [row(d, c="10", adj_c="8.0000004") for d in days])
    listing = {"X": ListingPeriod("X", date(2019, 12, 4), date(2019, 12, 10), source_url="u")}
    corr = {
        "X": (
            PriceCorrection(
                "X", date(2019, 12, 5), date(2019, 12, 6), CorrectionMethod.USE_RAW, "PM-T", "测试"
            ),
        )
    }
    data = loader(raw, listing, corr).load("X")
    assert data.history.first_day == date(2019, 12, 4)
    assert data.history.last_day == date(2019, 12, 10)
    by = {b.day: b.adj_close for b in data.history.bars}
    assert by[date(2019, 12, 4)] == 8.0  # 量化
    assert by[date(2019, 12, 5)] == 10.0 and by[date(2019, 12, 6)] == 10.0  # 更正
    assert data.ranges.left_end is LeftEnd.LISTING_START
    assert data.ranges.right_end is RightEnd.LISTING_END
    assert data.missing_days == ()


class FakeReference:
    library, version, calendar_name = "fake", "0", "XNYS"

    def __init__(self, days: tuple[date, ...]) -> None:
        self.days = days

    def sessions(self, start: date, end: date) -> tuple[date, ...]:
        return tuple(d for d in self.days if start <= d <= end)


class NullWriter:
    def write(self, result, out_dir: Path) -> list[Path]:
        return []


def test_compare_calendars_reports_both_sides() -> None:
    a = (date(2019, 1, 2), date(2019, 1, 3), date(2019, 1, 4))
    b = (date(2019, 1, 2), date(2019, 1, 4), date(2019, 1, 7))
    assert compare_calendars(a, b, date(2019, 1, 1), date(2019, 1, 31)) == (
        (date(2019, 1, 3),),
        (date(2019, 1, 7),),
    )


def test_check_calendar_service_detects_difference(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    ld = loader(raw)
    ref = FakeReference(tuple(d for d in SPY_DAYS if d != date(2019, 12, 24)))
    result, _ = CheckCalendarService(ld, ref, NullWriter()).run(date(2026, 10, 14), tmp_path)
    assert result.only_in_proxy == (date(2019, 12, 24),) and not result.identical
    same, _ = CheckCalendarService(ld, FakeReference(tuple(SPY_DAYS)), NullWriter()).run(
        date(2026, 10, 14), tmp_path
    )
    assert same.identical and same.proxy_file_sha256
