"""T09-E2 夹具比较：只做日序号与类型转换，不推导期望值。"""

from decimal import Decimal
from typing import Any

from stock_risk.domain.evaluation.h1 import find_f_ref, judge_h1
from stock_risk.domain.evaluation.method_series import MethodSeries
from stock_risk.domain.evaluation.outcomes import Caliber, classify
from stock_risk.domain.evaluation.params import PROPOSED_EXPLORATION
from stock_risk.domain.evaluation.series import EvalSeries
from stock_risk.domain.evaluation.windows import (
    RecoveryStatus,
    WindowParams,
    h_window,
    recovery_ref,
    s2_window,
)
from tests.unit.evaluation.helpers import load_fixtures, make_method, run_case

V = PROPOSED_EXPLORATION
WINDOW = WindowParams(
    V.rho, V.y_cap, V.y_max, V.n_min
)  # 18 号约定：ρ 0.5、Y_cap 60、Y_max 252、n_min 10


def b_case(case_id: str) -> dict[str, Any]:
    return load_fixtures()["B"][case_id]


def idx(series: EvalSeries, day: Any) -> int | None:
    return series.index(day) if day is not None else None


def dec(value: str | None) -> Decimal | None:
    return Decimal(value) if value is not None else None


def methods(series: EvalSeries, case: dict[str, Any]) -> dict[str, MethodSeries]:
    return {name: make_method(series, name, spec) for name, spec in case["methods"].items()}


def assert_outcomes(case_id: str, key: str, caliber: Caliber) -> None:
    case = b_case(case_id)
    series, chain = run_case(case)
    for name, method in methods(series, case).items():
        expected = case[key][name]
        assert len(expected) == len(chain.events), (case_id, name)
        for n, (event, exp) in enumerate(zip(chain.events, expected, strict=True)):
            o = classify(series, method, event, caliber)
            got = {
                "class": o.outcome_class.value,
                "S1": o.s1,
                "S1w": o.s1w,
                "f0": idx(series, o.f0),
                "U_subtype": o.u_subtype.value if o.u_subtype else None,
                "pre_peak_in_alert": o.pre_peak_in_alert,
                "first_start_day": idx(series, o.first_start_day),
                "first_start_drawdown": o.first_start_drawdown,
                "not_in_alert_days_after_first": o.not_in_alert_days_after_first,
                "trough_day_in_alert": o.trough_day_in_alert,
                "trough_day_release": o.trough_day_release,
            }
            want = {**exp, "S1": dec(exp["S1"]), "S1w": dec(exp["S1w"])}
            want["first_start_drawdown"] = dec(exp["first_start_drawdown"])
            assert got == want, (case_id, name, n, got, want)


def assert_h1(case_id: str) -> None:
    case = b_case(case_id)
    series, chain = run_case(case)
    refs = [find_f_ref(series, e) for e in chain.events]
    for name, method in methods(series, case).items():
        expected = case["expected_h1"][name]
        assert len(expected) == len(chain.events), (case_id, name)
        for n, (event, exp) in enumerate(zip(chain.events, expected, strict=True)):
            r = judge_h1(series, method, event, refs[n])
            got = {"F_ref": idx(series, r.f_ref), "result": r.status.value}
            if "path" in exp:
                got["path"] = r.path
            assert got == exp, (case_id, name, n, got, exp)
            if r.f_ref is None:
                assert r.reasons, "F_ref 为空时须给出原因代码"


def assert_s2(case_id: str) -> None:
    case = b_case(case_id)
    series, chain = run_case(case)
    for name, method in methods(series, case).items():
        for n, (event, exp) in enumerate(zip(chain.events, case["expected_s2"][name], strict=True)):
            if exp.get("not_evaluated"):
                assert not event.is_main, (case_id, n)
                continue
            rec = recovery_ref(series, chain, event, WINDOW)
            res = s2_window(series, method, chain, event, rec, WINDOW)
            w = res.window
            got = {
                "R_E": idx(series, rec.day),
                "window": None if w.empty else [idx(series, w.start), idx(series, w.end)],
                "length": w.length,
                "evaluable": w.evaluable,
                "alert_evaluable": w.alert_evaluable,
                "main_included": w.main_included,
                "S2": w.value,
                "s_main_included": res.s_main_included,
                "class": w.window_class.value if w.window_class else None,
                "plus_length": w.plus_length,
                "plus_included": w.plus_included,
                "S2_plus": w.value_plus,
                "s_plus_included": res.s_plus_included,
            }
            want = {**exp, "S2": dec(exp["S2"]), "S2_plus": dec(exp["S2_plus"])}
            assert got == want, (case_id, name, n, got, want)


def assert_h(case_id: str) -> None:
    case = b_case(case_id)
    series, chain = run_case(case)
    for name, method in methods(series, case).items():
        for n, (event, exp) in enumerate(zip(chain.events, case["expected_h"][name], strict=True)):
            if exp.get("not_evaluated"):
                assert not event.is_main, (case_id, n)
                continue
            rec = recovery_ref(series, chain, event, WINDOW)
            assert rec.status is not RecoveryStatus.FOUND
            w = h_window(series, method, chain, event, rec, WINDOW)
            got: dict[str, Any] = {
                "R_E": {"R_E": idx(series, rec.day), "status": rec.status.value},
                "empty": w.empty,
            }
            if not w.empty:
                got |= {
                    "window_days": [idx(series, w.start), idx(series, w.end)],
                    "length": w.length,
                    "evaluable": w.evaluable,
                    "alert_evaluable": w.alert_evaluable,
                    "complete": w.complete,
                    "main_included": w.main_included,
                    "h": w.value,
                    "class": w.window_class.value if w.window_class else None,
                    "plus_included": w.plus_included,
                    "h_plus": w.value_plus,
                }
            want = dict(exp)
            for key in ("h", "h_plus"):
                if key in want:
                    want[key] = dec(want[key])
            assert got == want, (case_id, name, n, got, want)
