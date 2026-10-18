"""T09-C1 C-05：解除后不利条件分别判定（期望值由产品经理在任务卡中推导）。

价格构造：C_r = 100。A_r 可得的用例在 r 之前放 20 日开 = 收 = 100、高 102.5、低 97.5 的日线，
TR 每日 5，r 为第 21 条日线，ATR20(r) = 5；A_r 不可得的用例在 r 之前只放 5 条日线（不足 21 条）。
窗口内各日开 = 收，高低为收 ± 0.5。
"""

from datetime import date
from decimal import Decimal

import pytest

from stock_risk.domain.evaluation.release_windows import AdverseThresholds, post_signal_window
from tests.unit.evaluation.helpers import make_series

D = Decimal
TH = AdverseThresholds(D("0.2"), D("3"))


def _flat(close: str) -> list[str]:
    return [close, f"{D(close) + D('0.5'):.6f}", f"{D(close) - D('0.5'):.6f}", close]


def _window(closes: list[str], *, atr: bool, q: int, l_ep: str | None, th=TH):  # type: ignore[no-untyped-def]
    pre = 20 if atr else 5
    bars = [["100.000000", "102.500000", "97.500000", "100.000000"]] * pre
    bars.append(["100.000000", "102.500000", "97.500000", "100.000000"])  # r
    bars += [_flat(f"{D(c):.6f}") for c in closes]
    series = make_series({"bars": bars})
    days = series.calendar.days
    w = post_signal_window(series, days[pre], q, D(l_ep) if l_ep else None, th)
    return w, days[pre + 1 :]


def test_case1_atr_unavailable_pct_hits_first_day_uncertain() -> None:
    w, d = _window(["98", "70", "72"], atr=False, q=3, l_ep="95")
    assert w.a_r is None and w.complete
    assert w.adverse is True
    assert w.known_criteria == ("PCT", "L_EP")
    assert w.first_adverse_day is None and w.first_criteria is None
    assert {"FIRST_DAY_UNCERTAIN", "ATR_UNAVAILABLE"} <= w.reasons


def test_case2_first_day_hit_with_atr_unavailable() -> None:
    w, d = _window(["70", "70", "72"], atr=False, q=3, l_ep="95")
    assert w.adverse is True and w.first_adverse_day == d[0]
    assert w.known_criteria == w.first_criteria == ("PCT", "L_EP")
    assert "ATR_UNAVAILABLE" in w.reasons and "FIRST_DAY_UNCERTAIN" not in w.reasons


def test_case3_atr_unavailable_nothing_known_true() -> None:
    w, _ = _window(["99", "98", "97"], atr=False, q=3, l_ep="95")
    assert w.adverse is None and "ATR_UNAVAILABLE" in w.reasons
    assert w.known_criteria is None and w.first_criteria is None


def test_case4_l_ep_unavailable_pct_and_atr_hit() -> None:
    w, d = _window(["70", "72", "75"], atr=True, q=3, l_ep=None)
    assert w.a_r == D(5)
    assert w.adverse is True and w.first_adverse_day == d[0]
    assert w.first_criteria == ("PCT", "ATR")
    assert "L_EP_UNAVAILABLE" in w.reasons


def test_case5_no_thresholds() -> None:
    w, _ = _window(["70", "72", "75"], atr=True, q=3, l_ep="95", th=None)
    assert w.adverse is None and "NO_THRESHOLDS" in w.reasons
    assert w.known_criteria is None and w.first_criteria is None and w.first_adverse_day is None


def test_case6b_incomplete_window_known_hit_is_adverse() -> None:
    w, d = _window(["99", "70", "97"], atr=True, q=5, l_ep="95")
    assert not w.complete and w.observed == 3
    assert w.adverse is True and w.first_adverse_day == d[1]
    assert w.known_criteria == w.first_criteria == ("PCT", "ATR", "L_EP")
    assert "INCOMPLETE_WINDOW" in w.reasons


def test_case6a_incomplete_window_without_hit_is_unknown() -> None:
    """窗口不完整、三日三个条件都已知不成立：未知，原因含 INCOMPLETE_WINDOW
    （T09-E5F 按 C1-01 裁定恢复 T09-C1 用例 6 第一例的断言；夹具 v1.4 的 B41e、B41f 同步）。"""
    w, _ = _window(["99", "98", "97"], atr=True, q=5, l_ep="95")
    assert not w.complete and "INCOMPLETE_WINDOW" in w.reasons
    assert w.adverse is None


@pytest.mark.parametrize("closes", [["99", "98", "97"], ["98", "70", "72"]])
def test_reasons_record_inputs_regardless_of_conclusion(closes: list[str]) -> None:
    """原因字段说明输入状况：缺失的输入无论结论是否确定都记入（G-V3-01）。"""
    w, _ = _window(closes, atr=False, q=5, l_ep=None)
    assert {"ATR_UNAVAILABLE", "L_EP_UNAVAILABLE", "INCOMPLETE_WINDOW"} <= w.reasons
    assert isinstance(w.signal_day, date)
