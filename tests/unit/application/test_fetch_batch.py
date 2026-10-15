"""T08d-1：fetch-batch 的额度记账、正常退出、429 退出、中断续传与封存（假传输层，不访问网络）。"""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from stock_risk.application.fetch_batch import BatchItem, FetchBatchService, QuotaLimits
from stock_risk.application.ports import PriceSourceError, RateLimitError
from stock_risk.domain.market_data import DailyBar, PriceHistory
from stock_risk.infrastructure.storage.csv_raw_repository import CsvRawDataRepository
from stock_risk.infrastructure.storage.fetch_batch_files import JsonQuotaStore, read_batch_list
from stock_risk.infrastructure.tiingo.client import TiingoPriceSource

FROM, TO = date(2008, 1, 2), date(2019, 12, 31)


def history(*days: date) -> PriceHistory:
    return PriceHistory(tuple(DailyBar(d, 10, 11, 9, 10, 100, 10, 11, 9, 10) for d in days), ())


class FakeSource:
    """按代码给出结果：PriceHistory、None（无数据）或异常。每次 fetch 计一次请求。"""

    name = "Fake"

    def __init__(self, results: dict, extra_requests: int = 0) -> None:
        self.results = results
        self.calls: list[str] = []
        self._count = 0
        self.extra = extra_requests

    @property
    def request_count(self) -> int:
        return self._count

    def fetch(self, ticker: str, start: date, end: date):  # type: ignore[no-untyped-def]
        self.calls.append(ticker)
        self._count += 1 + self.extra
        r = self.results.get(ticker, history(date(2019, 12, 30), date(2019, 12, 31)))
        if isinstance(r, BaseException) or (isinstance(r, type) and issubclass(r, BaseException)):
            raise r
        return r


class Clock:
    def __init__(self, t: datetime) -> None:
        self.t = t

    def __call__(self) -> datetime:
        return self.t


def items(*symbols: str) -> list[BatchItem]:
    return [BatchItem(s, s, FROM, TO) for s in symbols]


def service(out: Path, source: FakeSource, clock: Clock) -> FetchBatchService:
    return FetchBatchService(
        source, CsvRawDataRepository(out, title="选参集"), JsonQuotaStore(out), clock
    )


def quota(out: Path) -> dict:
    return json.loads((out / "_quota_state.json").read_text("utf-8"))


def status(out: Path) -> dict:
    return json.loads((out / "_fetch_status.json").read_text("utf-8"))


def test_fetch_and_resume_with_quota_bookkeeping(tmp_path: Path) -> None:
    clock = Clock(datetime(2026, 10, 14, 23, 0, tzinfo=UTC))
    src = FakeSource({"BBB": None})
    r = service(tmp_path, src, clock).run(items("AAA", "BBB", "CCC"), QuotaLimits())
    assert [o.symbol for o in r.done_now] == ["AAA", "BBB", "CCC"] and r.remaining == 0
    assert (tmp_path / "AAA.csv").exists() and not (tmp_path / "BBB.csv").exists()
    st = status(tmp_path)
    assert st["BBB"]["state"] == "no_data" and st["AAA"]["ok"]
    assert "AAA.csv" in (tmp_path / "MANIFEST.sha256").read_text("utf-8")
    assert quota(tmp_path) == {
        "day": "2026-10-14",
        "day_requests": 3,
        "month": "2026-10",
        "month_tickers": ["AAA", "BBB", "CCC"],
    }
    # 再次运行：成功与无数据都算完成，全部跳过，不再请求
    src2 = FakeSource({})
    r2 = service(tmp_path, src2, clock).run(items("AAA", "BBB", "CCC"), QuotaLimits())
    assert src2.calls == [] and r2.skipped_done == 3 and r2.cumulative_done == 3


def test_quota_rolls_over_day_and_month(tmp_path: Path) -> None:
    clock = Clock(datetime(2026, 10, 31, 23, 30, tzinfo=UTC))
    service(tmp_path, FakeSource({}), clock).run(items("A"), QuotaLimits())
    clock.t += timedelta(hours=1)  # 跨日、跨月（UTC）
    service(tmp_path, FakeSource({}), clock).run(items("A", "B"), QuotaLimits())
    q = quota(tmp_path)
    assert (q["day"], q["day_requests"], q["month"], q["month_tickers"]) == (
        "2026-11-01",
        1,
        "2026-11",
        ["B"],
    )


def test_per_day_limit_stops_normally(tmp_path: Path) -> None:
    clock = Clock(datetime(2026, 10, 14, 1, 0, tzinfo=UTC))
    src = FakeSource({}, extra_requests=1)  # 每个代码计 2 次请求（含一次重试）
    r = service(tmp_path, src, clock).run(items("A", "B", "C"), QuotaLimits(per_day=4))
    assert src.calls == ["A", "B"] and r.stop_reason == "per_day" and not r.rate_limited
    assert r.day_requests == 4 and r.remaining == 1 and r.estimated_runs == 1
    clock.t += timedelta(days=1)  # 次日继续
    r2 = service(tmp_path, FakeSource({}), clock).run(items("A", "B", "C"), QuotaLimits(per_day=4))
    assert r2.stop_reason is None and r2.cumulative_done == 3


def test_unique_per_month_and_max_symbols(tmp_path: Path) -> None:
    clock = Clock(datetime(2026, 10, 14, tzinfo=UTC))
    src = FakeSource({})
    r = service(tmp_path, src, clock).run(items("A", "B", "C"), QuotaLimits(unique_per_month=2))
    assert src.calls == ["A", "B"] and r.stop_reason == "unique_per_month"
    out2 = tmp_path / "b"
    src2 = FakeSource({})
    r2 = service(out2, src2, clock).run(items("A", "B", "C"), QuotaLimits(max_symbols=1))
    assert src2.calls == ["A"] and r2.stop_reason == "max_symbols"


def test_rate_limited_saves_progress_and_fails(tmp_path: Path) -> None:
    clock = Clock(datetime(2026, 10, 14, tzinfo=UTC))
    src = FakeSource({"B": RateLimitError("HTTP 429")})
    r = service(tmp_path, src, clock).run(items("A", "B", "C"), QuotaLimits())
    assert src.calls == ["A", "B"] and r.rate_limited and r.cumulative_done == 1
    assert "B" not in status(tmp_path)  # 未记为失败，下次重试
    r2 = service(tmp_path, FakeSource({}), clock).run(items("A", "B", "C"), QuotaLimits())
    assert r2.cumulative_done == 3


def test_failure_is_recorded_and_retried(tmp_path: Path) -> None:
    clock = Clock(datetime(2026, 10, 14, tzinfo=UTC))
    r = service(tmp_path, FakeSource({"A": PriceSourceError("HTTP 500")}), clock).run(
        items("A"), QuotaLimits()
    )
    assert [o.state for o in r.failed_now] == ["failed"] and status(tmp_path)["A"]["error"]
    src = FakeSource({})
    service(tmp_path, src, clock).run(items("A"), QuotaLimits())
    assert src.calls == ["A"]


def test_interrupt_keeps_progress(tmp_path: Path) -> None:
    clock = Clock(datetime(2026, 10, 14, tzinfo=UTC))
    src = FakeSource({"B": KeyboardInterrupt})
    with pytest.raises(KeyboardInterrupt):
        service(tmp_path, src, clock).run(items("A", "B", "C"), QuotaLimits())
    assert (tmp_path / "A.csv").exists() and quota(tmp_path)["day_requests"] == 2
    src2 = FakeSource({})
    r = service(tmp_path, src2, clock).run(items("A", "B", "C"), QuotaLimits())
    assert src2.calls == ["B", "C"] and r.cumulative_done == 3


class InterruptBeforeRequest(FakeSource):
    """在限频等待中被中断：请求尚未发出（request_count 不变）。"""

    def fetch(self, ticker: str, start: date, end: date):  # type: ignore[no-untyped-def]
        if ticker == "B":
            self.calls.append(ticker)
            raise KeyboardInterrupt
        return super().fetch(ticker, start, end)


def test_b04_month_ticker_only_counted_when_request_sent(tmp_path: Path) -> None:
    """Gemini 第二阶段 B-04：未发出请求就中断，不消耗当月新代码额度。"""
    clock = Clock(datetime(2026, 10, 14, tzinfo=UTC))
    with pytest.raises(KeyboardInterrupt):
        service(tmp_path, InterruptBeforeRequest({}), clock).run(items("A", "B"), QuotaLimits())
    q = quota(tmp_path)
    assert q["month_tickers"] == ["A"] and q["day_requests"] == 1


def test_b04_audit_dir_separates_no_data_from_missing(tmp_path: Path) -> None:
    from stock_risk.application.audit_raw_data import AuditRawDataService
    from stock_risk.application.market_data_loader import MarketDataLoader
    from stock_risk.infrastructure.storage.audit_report_writer import AuditReportFileWriter
    from stock_risk.infrastructure.storage.csv_raw_reader import CsvRawDataReader
    from stock_risk.infrastructure.storage.csv_raw_repository import confirmed_no_data

    clock = Clock(datetime(2026, 10, 14, tzinfo=UTC))
    out = tmp_path / "sel"
    service(out, FakeSource({"BBB": None}), clock).run(items("AAA", "BBB"), QuotaLimits())
    cal = tmp_path / "raw"
    service(cal, FakeSource({}), clock).run([BatchItem("SPY", "SPY", FROM, TO)], QuotaLimits())
    assert confirmed_no_data(out) == {"BBB"}

    class Members:
        def load(self):  # type: ignore[no-untyped-def]
            return [i.member() for i in items("AAA", "BBB", "CCC")]

    class NoConfig:
        def load(self) -> dict:
            return {}

    svc = AuditRawDataService(
        MarketDataLoader(CsvRawDataReader(out), NoConfig(), NoConfig()),
        Members(),
        NoConfig(),
        AuditReportFileWriter(),
        calendar_reader=CsvRawDataReader(cal),
        confirmed_no_data=confirmed_no_data(out),
    )
    report, _ = svc.run(date(2026, 10, 14), tmp_path / "audit")
    assert report.no_data == ["BBB"] and report.missing == ["CCC"]
    assert report.blocking_problems == ["缺少原始文件 1 个：CCC"]


def test_seal_on_returned_rows(tmp_path: Path) -> None:
    """数据源多返回的 2020 年行在写盘前被截断（第二道关）。"""
    clock = Clock(datetime(2026, 10, 14, tzinfo=UTC))
    src = FakeSource({"A": history(date(2019, 12, 31), date(2020, 1, 2))})
    service(tmp_path, src, clock).run(items("A"), QuotaLimits())
    assert "2020-" not in (tmp_path / "A.csv").read_text("utf-8")


def test_list_rejects_dates_after_seal(tmp_path: Path) -> None:
    p = tmp_path / "list.csv"
    p.write_bytes(b"symbol,source_ticker,from,to\nA,A,2008-01-02,2020-01-02\n")
    with pytest.raises(ValueError, match="不得晚于"):
        read_batch_list(p)
    p.write_bytes(b"symbol,source_ticker,from,to\na,a,2008-01-02,2019-12-31\n")
    assert read_batch_list(p) == [BatchItem("A", "A", FROM, TO)]
    with pytest.raises(ValueError, match="重复"):
        service(tmp_path, FakeSource({}), Clock(datetime.now(UTC))).run(
            items("A", "A"), QuotaLimits()
        )


def test_client_raises_rate_limit_after_three_429_and_counts_requests() -> None:
    def transport(req):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many", {}, io.BytesIO(b""))

    src = TiingoPriceSource("fake-key", transport=transport, sleep=lambda _: None)
    with pytest.raises(RateLimitError) as e:
        src.fetch("AAA", FROM, TO)
    assert src.request_count == 3 and "fake-key" not in str(e.value)
