"""18 号 B 组与 D53（T09 卡 E2）：每个夹具一个测试函数，期望值逐项取自夹具 JSON。

方法的逐日状态由夹具区段直接构造，不经引擎；H1 的 F_ref 经 D-21 的引擎函数计算。
B26 的掩码断言在 test_series.py（E1）。
"""

from decimal import Decimal

import pytest

from stock_risk.domain.evaluation.membership import (
    Membership,
    attribute,
    included_events,
    window_determinable,
)
from stock_risk.domain.evaluation.method_series import MethodSeries
from stock_risk.domain.evaluation.outcomes import Caliber, OutcomeClass, classify
from stock_risk.domain.evaluation.release_windows import AdverseThresholds, post_signal_window
from stock_risk.domain.evaluation.windows import recovery_ref, s2_window
from tests.unit.evaluation.e2_helpers import (
    WINDOW,
    assert_h,
    assert_h1,
    assert_outcomes,
    assert_s2,
    b_case,
    idx,
    methods,
)
from tests.unit.evaluation.helpers import load_fixtures, make_series, run_case


def test_B13_outcome_classes_s1_s1w_first_alert() -> None:
    assert_outcomes("B13", "expected_outcomes", Caliber.MAIN)


def test_B14_h1_start_on_fref_and_release_before() -> None:
    assert_h1("B14")


def test_B14n_h1_fref_absent_not_judged() -> None:
    assert_h1("B14n")


def test_B33_fref_anchored_to_own_peak() -> None:
    assert_h1("B33")


def test_B40_fref_equals_peak_by_gap() -> None:
    assert_h1("B40")


def test_B21_right_censored_prefix_complete() -> None:
    assert_h1("B21")
    assert_outcomes("B21", "expected_outcomes_provisional", Caliber.PROVISIONAL)
    # 主口径不纳入右端截尾事件：记 EXCLUDED 并给出原因，指标为空（不写成 0 或 1）
    case = b_case("B21")
    series, chain = run_case(case)
    for method in methods(series, case).values():
        o = classify(series, method, chain.events[0], Caliber.MAIN)
        assert o.outcome_class is OutcomeClass.EXCLUDED and o.reasons == {"RIGHT_DATA_END"}
        assert o.s1 is None and o.s1w is None


def test_B21m_missing_censor_h1_unknown() -> None:
    assert_h1("B21m")
    case = b_case("B21m")
    series, chain = run_case(case)
    for method in methods(series, case).values():
        for caliber in Caliber:
            o = classify(series, method, chain.events[0], caliber)
            assert o.outcome_class is OutcomeClass.EXCLUDED
            assert o.reasons == {"MISSING", "RIGHT_DATA_END", "UNEVALUABLE_MISSING"}
    # 事件的截尾与不可评估逐项比较夹具（v1.2，T09-E2M 补正 E2-01）
    expected = case["expected_events"][0]
    assert chain.events[0].censor == frozenset(expected["censor"])
    assert chain.events[0].unevaluable == frozenset(expected["unevaluable"])


def test_B15_s2_restart_truncated_and_supplemented() -> None:
    assert_s2("B15")


def test_B15e_s2_empty_window() -> None:
    assert_s2("B15e")


def test_B36_supplement_class_b() -> None:
    assert_s2("B36")


def test_B35_supplement_class_a() -> None:
    assert_s2("B35")


def test_B20a_h_tail_complete() -> None:
    assert_h("B20a")


def test_B20c_h_tail_insufficient_observation() -> None:
    assert_h("B20c")


def test_B20e_h_tail_empty() -> None:
    assert_h("B20e")


@pytest.mark.parametrize("sub", ["B41a", "B41b", "B41c", "B41d1", "B41d2", "B41e", "B41f"])
def test_B41_post_release_windows(sub: str) -> None:
    case = b_case("B41")[sub]
    series = make_series(case)
    p = case["params"]
    thresholds = AdverseThresholds(Decimal(p["y_pct"]), Decimal(p["y_atr"]))
    release = series.calendar.days[case["release_day"]]
    for q, key in ((p["Q"], "20"), (p["Q_L"], "60")):
        w = post_signal_window(series, release, q, Decimal(case["L_ep"]), thresholds)
        got = {
            "complete": w.complete,
            "adverse": w.adverse,
            "first_adverse_day": idx(series, w.first_adverse_day),
            # 夹具的 criteria 即首个不利日满足的条件（T09-C1 C-05 起字段名为 first_criteria）
            "criteria": list(w.first_criteria) if w.first_criteria is not None else None,
            "observed": w.observed,
            "A_r": w.a_r,
        }
        exp = case["expected"][key]
        assert got == {**exp, "A_r": Decimal(exp["A_r"])}, (sub, q, got, exp)


def test_B22_membership_by_peak_and_window_boundary() -> None:
    case = b_case("B22")
    series, chain = run_case(case)
    days = series.calendar.days
    none = MethodSeries("NONE", days, (None,) * len(days), (None,) * len(days), ())
    for sub in case["cases"]:
        member = Membership(tuple((days[a], days[b]) for a, b in sub["membership"]))
        peaks = [idx(series, e.p) for e in included_events(chain, member)]
        assert peaks == sub["included_event_peaks"], sub
        if "event1_S2_window" in sub:  # 未入选的事件 2 仍把事件 1 的窗口截短
            event = chain.events[0]
            w = s2_window(
                series, none, chain, event, recovery_ref(series, chain, event, WINDOW), WINDOW
            )
            assert [idx(series, w.window.start), idx(series, w.window.end)] == sub[
                "event1_S2_window"
            ]
            assert w.window.window_class is not None
            assert w.window.window_class.value == sub["event1_S2_class"]


def test_D53_two_segment_attribution_by_determinable_day() -> None:
    d53 = load_fixtures()["D"]["D53"]
    series, chain = run_case(load_fixtures()["A"][d53["use_bars_of"]])
    days = series.calendar.days
    none = MethodSeries("NONE", days, (None,) * len(days), (None,) * len(days), ())
    front_end = series.valid_ordinals[d53["front_end_day"]]
    assert front_end is not None
    for exp in d53["expected"]:
        event = chain.events[exp["event"] - 1]
        if exp["metric"] == "S1":  # S1 的可确定日 = 事件标签的可确定日（17 §2.2）
            assert event.label_determinable is not None
            det = series.valid_ordinals[series.index(event.label_determinable)]
        else:
            rec = recovery_ref(series, chain, event, WINDOW)
            window = s2_window(series, none, chain, event, rec, WINDOW).window
            det = window_determinable(series, chain, event, window)
        peak = series.valid_ordinals[series.index(event.p)]
        assert peak is not None
        # 夹具无缺价日：有效日序号与日序号相同
        assert det == exp["determinable"], exp
        assert attribute(det, peak, front_end).value == exp["segment"], exp
