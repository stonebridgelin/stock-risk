"""T09-C1 C-02：h 窗口被下一事件截短时，可确定日要等下一事件的标签；可确定日为空的指标记 NEITHER。

回归用例的期望值由产品经理给出（A04），构造用例按 17 §2.2 的公式计算，不取实现输出。
"""

import copy

import pytest

from stock_risk.domain.evaluation.membership import Segment, attribute, window_determinable
from stock_risk.domain.evaluation.method_series import MethodSeries
from stock_risk.domain.evaluation.windows import (
    RecoveryStatus,
    WindowStats,
    h_window,
    recovery_ref,
    s2_window,
)
from tests.unit.evaluation.e2_helpers import WINDOW
from tests.unit.evaluation.helpers import load_fixtures, run_case


@pytest.mark.parametrize(
    ("determinable", "peak", "expected"),
    [
        (None, 50, Segment.NEITHER),  # 峰值在前段，可确定日为空
        (None, 150, Segment.NEITHER),  # 峰值在后段，可确定日为空
        (100, 50, Segment.FRONT),  # 峰值在前段，可确定日不晚于分界日
        (101, 50, Segment.NEITHER),  # 峰值在前段，可确定日晚于分界日
        (160, 150, Segment.BACK),  # 峰值在后段，可确定日非空
    ],
)
def test_attribute_five_cases(determinable: int | None, peak: int, expected: Segment) -> None:
    assert attribute(determinable, peak, front_end=100) is expected


def _h_event0(case: dict) -> tuple:  # type: ignore[type-arg]
    series, chain = run_case(case)
    days = series.calendar.days
    none = MethodSeries("NONE", days, (None,) * len(days), (None,) * len(days), ())
    event = chain.events[0]
    rec = recovery_ref(series, chain, event, WINDOW)
    assert rec.status is RecoveryStatus.NOT_RECOVERED  # R_E 不存在，计算 h 窗口
    w = h_window(series, none, chain, event, rec, WINDOW)
    return series, chain, event, w


def test_A04_h_window_truncated_next_label_unknown() -> None:
    """A04：本事件标签可确定日 354；下一事件峰值 354，t_h = 353 < 109 + 252，截短；
    下一事件在数据终点仍未确认 → h 指标的可确定日为空，归段 NEITHER。"""
    series, chain, event, w = _h_event0(load_fixtures()["A"]["A04"])
    days = series.calendar.days
    assert event.label_determinable == days[354] and chain.events[1].p == days[354]
    assert w.truncated_by_next and w.target_end_ordinal == 353
    assert chain.events[1].label_determinable is None
    assert window_determinable(series, chain, event, w) is None
    assert attribute(None, 99, front_end=200) is Segment.NEITHER


def test_A04_extended_next_event_confirms_at_T() -> None:
    """构造：A04 之后接第 381 日，开 40、高 50.5、低 39.5、收 50。下一事件（峰值 354，C_p = 50）
    当日回到峰值，按 17 §2.2 (a) 结束并当日确认，T = 381；它的前序事件（事件 0）在 354 确认，
    不晚于 T。期望 h 指标的可确定日 = max(354, 381) = 381。"""
    case = copy.deepcopy(load_fixtures()["A"]["A04"])
    assert len(case["bars"]) == 381
    case["bars"].append(["40.000000", "50.500000", "39.500000", "50.000000"])
    series, chain, event, w = _h_event0(case)
    days = series.calendar.days
    t = 381
    assert chain.events[1].confirm == days[t]  # 构造的前提成立
    assert w.truncated_by_next
    expected = max(354, t)
    assert window_determinable(series, chain, event, w) == expected


def test_s2_truncated_windows_are_class_d() -> None:
    """S2：实际被下一事件截短（下一峰值 − 1 < Ω* 终点）的窗口都是 D 类（B 组夹具上逐个核对）。"""
    from stock_risk.domain.evaluation.windows import WindowClass
    from tests.unit.evaluation.helpers import make_method

    checked = 0
    for case_id in ("B15", "B15e", "B35", "B36", "B22"):
        case = load_fixtures()["B"][case_id]
        series, chain = run_case(case)
        for spec_name, spec in case.get("methods", {"X": {"spans": [], "events": []}}).items():
            method = make_method(series, spec_name, spec)
            for event in chain.events:
                if event.report_trough is None:
                    continue
                rec = recovery_ref(series, chain, event, WINDOW)
                if rec.status is not RecoveryStatus.FOUND:
                    continue
                w: WindowStats = s2_window(series, method, chain, event, rec, WINDOW).window
                later = [e for e in chain.events if e.s > event.s]
                if not later or w.target_end_ordinal is None:
                    continue
                nxt = series.valid_ordinals[series.index(min(later, key=lambda e: e.s).p)]
                assert nxt is not None
                if nxt - 1 < w.target_end_ordinal:
                    assert w.window_class is WindowClass.D and w.truncated_by_next
                    checked += 1
    assert checked >= 1
