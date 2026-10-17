"""T09-C1 C-03、C-04：分页未完成不记成功、下次重新获取；任何接口 401 / 403 停止整个运行。

假数据源，不访问网络；`probe-massive-paid` 与 `verify-massive-relay` 两个用例都测。
"""

from pathlib import Path

import pytest

from stock_risk.application.ports import QueryRejected
from stock_risk.domain.massive_followup import TickerPages
from tests.unit.application.test_probe_massive_paid import FakeSource, svc
from tests.unit.application.test_verify_massive_relay import (
    Q_ACTIVE,
    FakeRelay,
    _service,
)


class PartialActions(FakeSource):
    """拆股或分红的第一次请求分页未完成（只给第一页），之后完整。"""

    def __init__(self, which: str, partial: bool) -> None:
        super().__init__()
        self.which, self.partial = which, partial

    def splits(self, ticker, until):  # type: ignore[no-untyped-def]
        got = super().splits(ticker, until)
        if self.which == "splits" and self.partial:
            return TickerPages(got.rows[:0], 1, False)  # 第一页为空、还有下一页
        return got

    def dividends(self, ticker, until):  # type: ignore[no-untyped-def]
        got = super().dividends(ticker, until)
        if self.which == "dividends" and self.partial:
            return TickerPages(got.rows, 1, False)
        return got


@pytest.mark.parametrize("which", ["splits", "dividends"])
def test_paid_incomplete_actions_retried(tmp_path: Path, which: str) -> None:
    r1 = svc(tmp_path, PartialActions(which, partial=True), []).run()
    act = r1.summary["actions"]["MSFT"]
    assert act["status"] == "incomplete" and act[f"{which}_complete"] is False
    assert "self_split_adjust_vs_adjusted" not in act  # 不用于复权计算
    assert "actions_MSFT" in r1.summary["incomplete"] and "MSFT" in r1.remaining
    assert "actions_MSFT" in r1.incomplete
    # 第二次运行：只重新获取不完整的项，结果完整
    src2 = PartialActions(which, partial=False)
    r2 = svc(tmp_path, src2, []).run()
    assert sorted(src2.calls) == sorted(
        ["splits:MSFT", "dividends:MSFT", "splits:GM", "dividends:GM"]
    )
    act2 = r2.summary["actions"]["MSFT"]
    assert act2["status"] == "ok" and act2["splits"] == 1 and act2["dividends"] == 1
    assert r2.remaining == [] and r2.summary["incomplete"] == []
    saved = (tmp_path / "actions" / "MSFT_dividends.csv").read_text("utf-8")
    assert "2019-11-20" in saved


class RejectAt(FakeSource):
    def __init__(self, what: str, code: int) -> None:
        super().__init__()
        self.what, self.code = what, code

    def splits(self, ticker, until):  # type: ignore[no-untyped-def]
        if self.what == "splits":
            self._hit(f"splits:{ticker}")
            raise QueryRejected(self.code, "Unknown API Key")
        return super().splits(ticker, until)


@pytest.mark.parametrize("code", [401, 403])
def test_paid_any_401_403_stops_whole_run(tmp_path: Path, code: int) -> None:
    src = RejectAt("splits", code)
    r = svc(tmp_path, src, []).run()
    assert r.stopped == "plan_rejected"
    assert r.summary["plan_rejected"]["http_status"] == code
    assert r.summary["plan_rejected"]["what"] == "splits_MSFT"
    assert src.calls[-1] == "splits:MSFT"  # 此后不再发出任何请求
    src2 = FakeSource()
    r2 = svc(tmp_path, src2, []).run()
    assert r2.stopped == "plan_rejected" and src2.calls == []  # 第二次运行发出 0 次请求


@pytest.mark.parametrize("code", [401, 403])
@pytest.mark.parametrize("name", ["overview", f"list_{Q_ACTIVE.key}"])
def test_verify_non_price_401_403_stops_and_persists(tmp_path: Path, code: int, name: str) -> None:
    src = FakeRelay({name: QueryRejected(code, "Unknown API Key")})
    r = _service(tmp_path, src).run()
    assert r.stopped == "auth_rejected" and src.calls[-1] == name
    assert r.summary["auth_rejected"]["http_status"] == code
    again = FakeRelay()
    r2 = _service(tmp_path, again).run()
    assert r2.stopped == "auth_rejected" and again.calls == []


class PartialRelay(FakeRelay):
    """价格请求第一次分页未完成。"""

    def __init__(self, partial: bool) -> None:
        super().__init__()
        self.partial = partial

    def aggs(self, ticker, start, end, adjusted):  # type: ignore[no-untyped-def]
        got = super().aggs(ticker, start, end, adjusted)
        if self.partial and got is not None and self.calls[-1].startswith("prices_"):
            return TickerPages(got.rows, 1, False)
        return got


def test_verify_incomplete_prices_retried(tmp_path: Path) -> None:
    r1 = _service(tmp_path, PartialRelay(partial=True)).run()
    assert "prices_MSFT" in r1.remaining and "prices_MSFT" in r1.summary["incomplete"]
    src2 = PartialRelay(partial=False)
    r2 = _service(tmp_path, src2).run()
    assert src2.calls == ["prices_MSFT"]  # 只重新获取不完整的项
    assert r2.remaining == [] and r2.summary["incomplete"] == []
    summary = (tmp_path / "paid" / "verify" / "verify_prices_summary.csv").read_text("utf-8")
    assert "MSFT,ok" in summary
