"""T10-M0b 第 6 节：旧 Tiingo 工具拒绝复用未经验证的目录（F-06 ②）。

合成存储（临时目录中的 CsvRawDataRepository）与假数据源，不访问网络。每种情形断言请求次数；
拒绝的情形另断言目录中全部文件逐字节不变。
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from stock_risk.application.fetch_batch import (
    BatchItem,
    FetchBatchService,
    QuotaLimits,
    UnverifiedDirectory,
)
from stock_risk.application.ports import PriceSourceError, RateLimitError
from stock_risk.infrastructure.storage.csv_raw_repository import CsvRawDataRepository
from stock_risk.infrastructure.storage.fetch_batch_files import JsonQuotaStore
from tests.unit.application.test_fetch_batch import Clock, FakeSource, history

FROM, TO = date(2008, 1, 2), date(2019, 12, 31)
CLOCK = Clock(datetime(2026, 10, 17, 10, 0, tzinfo=UTC))
LEDGER = "_request_ledger.json"


class NamedSource(FakeSource):
    def __init__(self, results: dict, name: str = "Fake") -> None:
        super().__init__(results)
        self.name = name


def item(sym: str, start: date = FROM, end: date = TO, ticker: str | None = None) -> BatchItem:
    return BatchItem(sym, ticker or sym, start, end)


def run(out: Path, src: FakeSource, items: list[BatchItem], limits: QuotaLimits | None = None):  # type: ignore[no-untyped-def]
    svc = FetchBatchService(
        src, CsvRawDataRepository(out, title="选参集"), JsonQuotaStore(out), CLOCK
    )
    return svc.run(items, limits or QuotaLimits())


def snapshot(out: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(out.iterdir()) if p.is_file()}


def assert_rejected(out: Path, src: FakeSource, items: list[BatchItem], *words: str) -> None:
    before = snapshot(out)
    n = src.request_count
    with pytest.raises(UnverifiedDirectory) as e:
        run(out, src, items)
    assert src.request_count == n  # 0 次请求
    assert snapshot(out) == before  # 目录中的任何文件都不改动
    for w in words:
        assert w in str(e.value), (w, str(e.value))


def ledger(out: Path) -> dict:
    return json.loads((out / LEDGER).read_text("utf-8"))


def test_new_directory_writes_ledger_with_hashes(tmp_path: Path) -> None:
    src = FakeSource({"BBB": None})
    r = run(tmp_path, src, [item("AAA"), item("BBB")])
    assert src.request_count == 2 and r.remaining == 0
    led = ledger(tmp_path)
    assert led["source"] == "Fake"
    a = led["items"]["AAA"]
    assert (a["result"], a["source_ticker"], a["start"], a["end"]) == (
        "ok",
        "AAA",
        "2008-01-02",
        "2019-12-31",
    )
    assert set(a["sha256"]) == {"price", "actions"} and all(
        len(v) == 64 for v in a["sha256"].values()
    )
    assert led["items"]["BBB"]["result"] == "no_data" and "sha256" not in led["items"]["BBB"]


def test_old_directory_without_ledger_rejected(tmp_path: Path) -> None:
    run(tmp_path, FakeSource({}), [item("AAA")])
    (tmp_path / LEDGER).unlink()  # 模拟旧目录：有状态与数据文件，没有请求记录
    assert_rejected(tmp_path, FakeSource({}), [item("AAA")], "旧目录没有请求记录", "新目录")


def test_range_widened_rejected(tmp_path: Path) -> None:
    """ChatGPT 的复现例：先请求 2012 年，再请求 2008-01-02—2019-12-31。"""
    run(tmp_path, FakeSource({}), [item("AAA", date(2012, 1, 1), date(2012, 12, 31))])
    assert_rejected(tmp_path, FakeSource({}), [item("AAA")], "AAA", "start")


def test_source_ticker_changed_rejected(tmp_path: Path) -> None:
    run(tmp_path, FakeSource({}), [item("AAA")])
    assert_rejected(tmp_path, FakeSource({}), [item("AAA", ticker="AAA.X")], "source_ticker")


def test_source_name_changed_with_all_new_symbols_rejected(tmp_path: Path) -> None:
    run(tmp_path, FakeSource({}), [item("AAA")])
    assert_rejected(tmp_path, NamedSource({}, "Other"), [item("NEW1"), item("NEW2")], "数据源名称")


@pytest.mark.parametrize(
    ("damage", "word"),
    [
        (lambda d: (d / "AAA.csv").write_bytes((d / "AAA.csv").read_bytes()[:-20]), "SHA-256"),
        (lambda d: (d / "AAA.csv").unlink(), "缺失"),
        (lambda d: (d / "AAA.actions.csv").unlink(), "缺失"),
        (
            lambda d: (d / "AAA.actions.csv").write_text(
                "date,type,value\n2019-01-02,dividend,1\n", "utf-8"
            ),
            "SHA-256",
        ),
    ],
    ids=["price_truncated", "price_deleted", "actions_deleted", "actions_changed"],
)
def test_ok_file_damaged_rejected(tmp_path: Path, damage, word: str) -> None:  # type: ignore[no-untyped-def]
    run(tmp_path, FakeSource({}), [item("AAA")])
    damage(tmp_path)
    assert_rejected(tmp_path, FakeSource({}), [item("AAA")], "AAA", word)


def test_failed_without_files_retried(tmp_path: Path) -> None:
    run(tmp_path, FakeSource({"AAA": PriceSourceError("boom")}), [item("AAA")])
    assert ledger(tmp_path)["items"]["AAA"]["result"] == "failed"
    src = FakeSource({})
    run(tmp_path, src, [item("AAA")])
    assert src.request_count == 1
    rec = ledger(tmp_path)["items"]["AAA"]
    assert rec["result"] == "ok" and set(rec["sha256"]) == {"price", "actions"}


def test_pending_after_429_retried(tmp_path: Path) -> None:
    r = run(tmp_path, FakeSource({"AAA": RateLimitError("429")}), [item("AAA")])
    assert r.rate_limited and ledger(tmp_path)["items"]["AAA"]["result"] == "pending"
    src = FakeSource({})
    run(tmp_path, src, [item("AAA")])
    assert src.request_count == 1 and ledger(tmp_path)["items"]["AAA"]["result"] == "ok"


def test_failed_with_price_file_rejected(tmp_path: Path) -> None:
    run(tmp_path, FakeSource({"AAA": PriceSourceError("boom")}), [item("AAA")])
    (tmp_path / "AAA.csv").write_text("date\n", "utf-8")
    assert_rejected(tmp_path, FakeSource({}), [item("AAA")], "AAA", "来历不明")


def test_unrecorded_symbol_with_price_file_rejected(tmp_path: Path) -> None:
    run(tmp_path, FakeSource({}), [item("AAA")])
    (tmp_path / "BBB.csv").write_text("date\n", "utf-8")
    assert_rejected(tmp_path, FakeSource({}), [item("BBB")], "BBB", "未经验证")


def test_consistent_ledger_skips_done(tmp_path: Path) -> None:
    run(tmp_path, FakeSource({}), [item("AAA")])
    src = FakeSource({})
    r = run(tmp_path, src, [item("AAA")])
    assert src.request_count == 0 and r.skipped_done == 1 and r.remaining == 0


def test_no_data_skipped_without_files(tmp_path: Path) -> None:
    run(tmp_path, FakeSource({"AAA": None}), [item("AAA")])
    src = FakeSource({})
    r = run(tmp_path, src, [item("AAA")])
    assert src.request_count == 0 and r.skipped_done == 1 and r.cumulative_done == 1


def test_only_new_symbols_requested_and_appended(tmp_path: Path) -> None:
    run(tmp_path, FakeSource({}), [item("AAA")])
    src = FakeSource({})
    run(tmp_path, src, [item("AAA"), item("CCC")])
    assert src.calls == ["CCC"]
    assert set(ledger(tmp_path)["items"]) == {"AAA", "CCC"}


def test_new_symbol_with_orphan_actions_file_rejected(tmp_path: Path) -> None:
    run(tmp_path, FakeSource({}), [item("AAA")])
    (tmp_path / "DDD.actions.csv").write_text("date,type,value\n", "utf-8")
    assert_rejected(tmp_path, FakeSource({}), [item("DDD")], "DDD")


def test_no_data_with_actions_file_rejected(tmp_path: Path) -> None:
    run(tmp_path, FakeSource({"AAA": None}), [item("AAA")])
    (tmp_path / "AAA.actions.csv").write_text("date,type,value\n", "utf-8")
    assert_rejected(tmp_path, FakeSource({}), [item("AAA")], "AAA", "no_data")


def _pending_with_stale_no_data(out: Path) -> None:
    """请求记录为未完成，旧 _fetch_status.json 却写着 no_data，目录中没有文件。"""
    run(out, FakeSource({"AAA": None}), [item("AAA")])
    led = ledger(out)
    led["items"]["AAA"]["result"] = "pending"
    (out / LEDGER).write_text(json.dumps(led), "utf-8")
    assert json.loads((out / "_fetch_status.json").read_text("utf-8"))["AAA"]["state"] == "no_data"


def test_pending_not_overridden_by_old_status(tmp_path: Path) -> None:
    _pending_with_stale_no_data(tmp_path)
    src = FakeSource({"AAA": history(date(2019, 12, 31))})
    r = run(tmp_path, src, [item("AAA")])
    assert src.request_count == 1 and r.remaining == 0
    assert ledger(tmp_path)["items"]["AAA"]["result"] == "ok"


def test_pending_counted_remaining_when_quota_exhausted(tmp_path: Path) -> None:
    _pending_with_stale_no_data(tmp_path)
    quota = json.loads((tmp_path / "_quota_state.json").read_text("utf-8"))
    src = FakeSource({})
    r = run(tmp_path, src, [item("AAA")], QuotaLimits(per_day=quota["day_requests"]))
    assert src.request_count == 0 and r.stop_reason == "per_day"
    assert r.cumulative_done == 0 and r.remaining == 1  # 仍计为未完成
