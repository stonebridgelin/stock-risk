"""T08d-2c 第 2、3 节：中转服务核验（假数据源，不访问网络，假密钥）。

- 试探 7 次请求的状态码、字段名、条数与封存丢弃；
- 价格逐日对账与参考数据比对的期望值按定义手算；
- 401 / 403 立即停止全部运行，再次运行不发请求；404 记为不可用，依赖它的第 3 节部分跳过；
- 429 用尽保存进度，续传完成；
- 输出带 source_host，不含 2020 年及以后的日期、不含密钥；
- 命令行：输入 yes 之前不发请求。
"""

from __future__ import annotations

import argparse
import builtins
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from stock_risk.application.ports import QueryRejected, RateLimitError
from stock_risk.application.verify_massive_relay import VerifyMassiveRelayService
from stock_risk.config.settings import load_settings
from stock_risk.domain.massive_followup import TickerPages
from stock_risk.domain.massive_paid import Bar, ReferenceBar
from stock_risk.domain.massive_probe import ListQuery
from stock_risk.domain.massive_verify import compare_daily, compare_snapshot, quantile7
from stock_risk.infrastructure.storage.massive_verify_files import MassiveVerifyFiles
from stock_risk.interfaces.cli import massive as cli

D = date
KEY = "fake-relay-key-999"
HOST = "relaynex.vip"
Q_ACTIVE = ListQuery(D(2019, 6, 3), "XNYS", "CS", True)
Q_INACTIVE = ListQuery(D(2019, 6, 3), "XNYS", "CS", False)


def _bar(day: date, close: float, volume: float = 100.0) -> Bar:
    return Bar(day, close, close, close, close, volume, None, None)


def _agg(day: date, close: float, volume: float = 100.0) -> dict[str, Any]:
    """Aggregates 的一行：t 为美东零点（UTC 05:00）的毫秒时间戳。"""
    ts = datetime(day.year, day.month, day.day, 5, tzinfo=UTC).timestamp() * 1000
    return {"t": int(ts), "o": close, "h": close, "l": close, "c": close, "v": volume}


# ---- 纯函数：手算期望 ----


def test_compare_daily_hand_computed() -> None:
    d1, d2, d3, d4, d5 = D(2015, 1, 5), D(2015, 1, 6), D(2015, 1, 7), D(2015, 1, 8), D(2015, 1, 9)
    relay = [
        _bar(D(2009, 12, 31), 1.0),  # 区间外，不计
        _bar(d1, 10.0, 110.0),
        _bar(d2, 10.02, 100.0),
        _bar(d3, 9.995, 90.0),
        _bar(d4, 10.0),
    ]
    tiingo = [ReferenceBar(d, 10.0, 100.0) for d in (d1, d2, d3, d5)]
    rows, s = compare_daily(relay, tiingo, [d1, d2, d3, d4])
    assert [r["side"] for r in rows] == ["both", "both", "both", "relay_only", "tiingo_only"]
    assert (s["common_days"], s["relay_only_dates"], s["tiingo_only_dates"]) == (
        3,
        "2015-01-08",
        "2015-01-09",
    )
    # 差值 0、0.02、−0.005：超过 0.01 的 1 天，占 1/3；中位数 0；按绝对值排序
    assert s["max_abs_close_diff"] == 0.02 and s["days_over_0_01"] == 1
    assert s["share_over_0_01"] == 0.333333 and s["median_close_diff"] == 0.0
    assert s["top5_abs_diff"] == "2015-01-06:0.02|2015-01-07:-0.005|2015-01-05:0.0"
    # 成交量相对差 0.1、0、−0.1：中位数 0；绝对值 [0, 0.1, 0.1] 的第 95 百分位 = 0.1
    assert s["volume_rel_diff_median"] == 0.0 and s["volume_abs_rel_diff_p95"] == 0.1
    assert rows[4]["in_spy_calendar"] is False and rows[0]["in_spy_calendar"] is True


def test_quantile7() -> None:
    assert quantile7([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5  # h = 1.5
    assert quantile7([], 0.5) is None


def test_compare_snapshot_hand_computed() -> None:
    def row(t: str, cik: str = "1", after: bool = False) -> dict[str, Any]:
        return {
            "ticker": t,
            "primary_exchange": "XNYS",
            "type": "CS",
            "active": "true",
            "cik": cik,
            "composite_figi": "",
            "list_date": "",
            "delisted": "",
            "delisted_after_2019": "true" if after else "false",
        }

    official = {"q": [row("A"), row("B"), row("C")]}
    relay = {"q": [row("A"), row("B", cik="2"), row("D", after=True)]}
    s, ex = compare_snapshot(D(2019, 6, 3), official, relay)
    assert (s["official_rows"], s["relay_rows"], s["only_official"], s["only_relay"]) == (
        3,
        3,
        1,
        1,
    )
    assert s["only_either_share"] == 0.666667  # (1 + 1) / 3
    assert s["rows_with_field_diff"] == 1 and s["diff_cik"] == 1 and s["diff_type"] == 0
    kinds = [(e["kind"], e["ticker"], e["field"]) for e in ex]
    assert kinds == [
        ("only_official", "C", ""),
        ("only_relay", "D", ""),
        ("field_diff", "B", "cik"),
    ]


# ---- 用例 ----


class FakeRelay:
    def __init__(self, fail: dict[str, Any] | None = None) -> None:
        self.fail = fail or {}  # 名称 → 异常或 None（404）
        self.calls: list[str] = []
        self.source_host = HOST

    @property
    def request_count(self) -> int:
        return len(self.calls)

    def _hit(self, name: str) -> None:
        self.calls.append(name)
        if name in self.fail and isinstance(self.fail[name], Exception):
            exc = self.fail[name]
            if isinstance(exc, RateLimitError):
                del self.fail[name]  # 只限频一次
            raise exc

    def aggs(self, ticker: str, start: date, end: date, adjusted: bool) -> TickerPages | None:
        if start == D(2010, 1, 1):  # 第 3.1 节的价格对账
            name = f"prices_{ticker}"
        else:
            name = "aggs_adj" if adjusted else "aggs_raw"
        self._hit(name)
        if self.fail.get(name, "x") is None:
            return None
        rows = [_agg(D(2015, 1, 5), 10.0), _agg(D(2015, 1, 6), 10.02), _agg(D(2020, 1, 2), 11.0)]
        return TickerPages(rows, 1, True)

    def splits(self, ticker: str, until: date) -> TickerPages:
        self._hit("splits")
        rows = [
            {"execution_date": "2014-06-09", "split_from": 1, "split_to": 7},
            {"execution_date": "2020-08-31", "split_from": 1, "split_to": 4},  # 封存丢弃
        ]
        return TickerPages(rows, 1, True)

    def dividends(self, ticker: str, until: date) -> TickerPages:
        self._hit("dividends")
        return TickerPages([{"ex_dividend_date": "2019-03-14", "cash_amount": 0.39}], 1, True)

    def list_tickers_pages(self, query: ListQuery, max_pages: int) -> TickerPages:
        name = "list_active" if query.market else f"list_{query.key}"
        if query.market and not query.active:
            name = "list_inactive"
        self._hit(name)
        rows = [
            {"ticker": "A", "primary_exchange": "XNYS", "type": "CS", "active": query.active,
             "cik": "1", "delisted_utc": "2021-01-04T00:00:00Z"},
            {"ticker": "Z", "list_date": "2020-03-02"},  # 封存丢弃
        ]  # fmt: skip
        return TickerPages(rows, 1, True)

    def overview(self, ticker: str, day: date) -> dict[str, Any] | None:
        self._hit("overview")
        return {"ticker": "MSFT", "sic_code": "7372"}


OFFICIAL = {
    Q_ACTIVE.key: [
        {"ticker": "A", "primary_exchange": "XNYS", "type": "CS", "active": "true", "cik": "1",
         "composite_figi": "", "list_date": "", "delisted": "", "delisted_after_2019": "true"},
    ],
    Q_INACTIVE.key: [],
}  # fmt: skip


def _service(tmp_path: Path, source: FakeRelay) -> VerifyMassiveRelayService:
    return VerifyMassiveRelayService(
        source=source,
        store=MassiveVerifyFiles(tmp_path / "paid", HOST),
        calendar=[D(2015, 1, 5), D(2015, 1, 6)],
        tickers=["MSFT"],
        tiingo_prices={"MSFT": [ReferenceBar(D(2015, 1, 5), 10.0, 100.0)]},
        official_lists=OFFICIAL,
        clock=lambda: 0.0,
        queries=[Q_ACTIVE, Q_INACTIVE],
    )


def _outputs(tmp_path: Path) -> str:
    return "".join(
        p.read_text("utf-8") for p in sorted((tmp_path / "paid").rglob("*")) if p.is_file()
    )


def test_full_run_outputs(tmp_path: Path) -> None:
    src = FakeRelay()
    r = _service(tmp_path, src).run()
    assert r.stopped is None and r.remaining == []
    names = ["aggs_raw", "aggs_adj", "splits", "dividends", "list_active", "list_inactive"]
    assert src.calls == [
        *names,
        "overview",
        "prices_MSFT",
        f"list_{Q_ACTIVE.key}",
        f"list_{Q_INACTIVE.key}",
    ]
    verify = tmp_path / "paid" / "verify"
    for f in ("coverage.csv", "verify_prices_summary.csv", "verify_reference_summary.csv",
              "verify_reference_examples.csv", "summary.json", "prices_MSFT.csv"):  # fmt: skip
        assert HOST in (verify / f).read_text("utf-8"), f
    seal = r.summary["seal"]
    # 试探：日线两次各丢 1 行、拆股丢 1 行、两种清单各丢 1 行；价格丢 1 行；32 种清单各丢 1 行
    assert seal["coverage_rows_dropped_after_2019"] == 5
    assert seal["price_rows_dropped_after_2019"] == 1
    assert seal["list_rows_dropped_list_date_after_2019"] == 2
    assert seal["list_rows_delisted_after_2019_masked"] == 2
    cov = (verify / "coverage.csv").read_text("utf-8")
    assert "200" in cov and "sic_code|ticker" in cov
    ref = (verify / "verify_reference_summary.csv").read_text("utf-8").splitlines()
    assert ref[1].startswith("2019-06-03,ok,1,2,0,1,")  # 官方 1 条；中转 2 条（A 与 inactive 的 A）
    text = _outputs(tmp_path)
    assert "2020-" not in text and "2021-" not in text and KEY not in text


@pytest.mark.parametrize("status", [401, 403])
def test_auth_rejection_stops_everything(tmp_path: Path, status: int) -> None:
    src = FakeRelay({"aggs_adj": QueryRejected(status, "Unknown API Key")})
    r = _service(tmp_path, src).run()
    assert r.stopped == "auth_rejected" and src.calls == ["aggs_raw", "aggs_adj"]
    assert r.summary["auth_rejected"]["http_status"] == status
    again = FakeRelay()
    r2 = _service(tmp_path, again).run()
    assert r2.stopped == "auth_rejected" and again.calls == []  # 再次运行不发请求


def test_not_found_marks_unavailable_and_skips(tmp_path: Path) -> None:
    src = FakeRelay({"aggs_raw": None})
    r = _service(tmp_path, src).run()
    assert r.stopped is None and "aggs_raw" in r.summary["unavailable"]
    assert "prices_MSFT" not in src.calls and f"list_{Q_ACTIVE.key}" in src.calls
    prices = (tmp_path / "paid" / "verify" / "verify_prices_summary.csv").read_text("utf-8")
    assert "MSFT,skipped_unavailable" in prices
    cov = (tmp_path / "paid" / "verify" / "coverage.csv").read_text("utf-8")
    assert "aggs_raw," in cov and "not_found,404" in cov


def test_rate_limit_saves_and_resumes(tmp_path: Path) -> None:
    src = FakeRelay({"prices_MSFT": RateLimitError("429")})
    r = _service(tmp_path, src).run()
    assert r.stopped == "rate_limited" and "prices_MSFT" in r.remaining
    src2 = FakeRelay()
    r2 = _service(tmp_path, src2).run()
    assert r2.stopped is None and src2.calls[0] == "prices_MSFT"  # 试探不重做
    assert r2.summary["requests_total"] == len(src.calls) + len(src2.calls)


# ---- 命令行：输入 yes 之前不发请求 ----


class _FakeService:
    source_host = HOST
    planned = {"coverage": 7, "prices": 27, "lists": 32}

    def __init__(self) -> None:
        self.ran = False

    def run(self) -> Any:
        self.ran = True
        raise KeyboardInterrupt  # 只验证确认流程


@pytest.mark.parametrize(("answer", "ran"), [("no", False), ("", False), ("yes", True)])
def test_cli_requires_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    answer: str, ran: bool,
) -> None:  # fmt: skip
    svc = _FakeService()
    monkeypatch.setattr(cli, "load_settings", lambda: load_settings(tmp_path))
    monkeypatch.setattr(cli, "build_verify_massive_relay_service", lambda *a, **k: svc)
    monkeypatch.setattr(builtins, "input", lambda prompt="": answer)
    code = cli._cmd_verify_massive_relay(argparse.Namespace(per_minute=None))  # noqa: SLF001
    assert svc.ran is ran and code == (130 if ran else 1)
    assert f"将要连接的主机：{HOST}" in capsys.readouterr().out
