"""T13-H1C：H1 未知的甲、乙分类（17 号第 4 节第 1.0–1.3 节、第 2 节；第 10 节第 21、21a、21b 项）。

期望值全部取自任务卡第 2 节与 17 号第 4 节，或手算；不从实现结果回填。
单位层夹具直接构造 SevereEvent、FRef、H1Result，不经价格序列。上限数值都是测试值，不是拟定值。
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.evaluation.event_model import (
    EventKind,
    EventParameters,
    EventSetSpec,
    PendingInterval,
    SevereEvent,
)
from stock_risk.domain.evaluation.events import build_events
from stock_risk.domain.evaluation.h1 import FRef, H1Result, H1Status, find_f_ref, judge_h1
from stock_risk.domain.evaluation.h1_classes import (
    CAPS_NOT_APPROVED,
    CapConclusion,
    Caps,
    Category,
    ClassBReason,
    DatasetRole,
    EventInput,
    H1ContractError,
    H1Unit,
    MethodKind,
    MethodState,
    UnitId,
    UnitKind,
    UnitResult,
    check_units,
    classify_h1,
    classify_reasons,
    expected_units,
    interval_unit,
    method_state,
    method_units,
    summarize,
)
from stock_risk.domain.evaluation.method_series import MethodSeries
from stock_risk.domain.evaluation.outcomes import exclusion_reasons
from stock_risk.domain.evaluation.series import EndReason, EvalSeries
from stock_risk.domain.market_data import DailyBar
from tests.unit.evaluation.e2_helpers import b_case, methods
from tests.unit.evaluation.helpers import (
    assert_case,
    load_fixtures,
    make_method,
    make_series,
    run_case,
    workdays,
)

SEC, ESET, M = "AAA", "E_pct", "R0"
DATA_END = date(2019, 12, 31)
FREF_DAY = date(2015, 3, 10)
S, P = date(2015, 3, 5), date(2015, 3, 2)
CONFIRMED = date(2015, 4, 1)
A, B = UnitResult.CLASS_A, UnitResult.CLASS_B
DEL, MIS, WARM, RIGHT = (
    Category.DELISTED,
    Category.MISSING,
    Category.WARMUP,
    Category.RIGHT_CENSORED,
)
FIXTURE_SHA = "408e9585036201708b7b48587244c7c2cde833b4b0a1b57f04757533d035ce86"


def ev(
    *,
    main: bool = False,
    prefix: bool = False,
    confirmed: bool = False,
    e: date | None = None,
    censor: tuple[str, ...] = (),
    unevaluable: tuple[str, ...] = (),
    dep: tuple[date, date, int] | None = None,
    s: date = S,
    p: date = P,
) -> SevereEvent:
    """未写明的字段取中性值（卡第 2.1 节）。已确认的事件 confirm = e = 2015-04-01。"""
    if confirmed:
        e = CONFIRMED
    return SevereEvent(
        s=s,
        p=p,
        c_p=Decimal("100"),
        e=e,
        reason="a" if confirmed else None,
        confirm=CONFIRMED if confirmed else None,
        flags=frozenset(),
        report_trough=date(2015, 3, 20) if confirmed else None,
        k_star=None if confirmed else date(2015, 3, 20),
        censor=frozenset(censor),
        unevaluable=frozenset(unevaluable),
        dependency_pending=PendingInterval(*dep) if dep is not None else None,
        label_determinable=None,
        cap_continuation=False,
        merges=0,
        is_main=main,
        is_prefix_complete=prefix,
        a_s=None,
    )


F_FOUND = FRef(True, FREF_DAY, "S", frozenset())
F_ABSENT = FRef(True, None, None, frozenset({"F_REF_ABSENT"}))


def unsearched(event: SevereEvent) -> FRef:
    return FRef(False, None, None, exclusion_reasons(event))


def judged(status: H1Status) -> H1Result:
    return H1Result(FREF_DAY, "S", status, frozenset(), Decimal("0.1"))


def classify(event: SevereEvent, f: FRef, j: H1Result | None = None, state=MethodState.COMPLETE):  # noqa: ANN001, ANN201
    return classify_h1(SEC, ESET, M, event, f, j, state, DATA_END)


def interval(event: SevereEvent, state=MethodState.COMPLETE):  # noqa: ANN001, ANN201
    return interval_unit(SEC, ESET, M, event, state, DATA_END)


# ---- 2.2 单位层（HC 组） ----


def test_HC01_main_fref_found_ok() -> None:
    u = classify(ev(main=True, confirmed=True), F_FOUND, judged(H1Status.OK))
    assert u.result is UnitResult.OK and u.kind is UnitKind.EVENT and u.as_of == P


def test_HC02_main_fref_found_violation() -> None:
    u = classify(ev(main=True, confirmed=True), F_FOUND, judged(H1Status.VIOLATION))
    assert u.result is UnitResult.VIOLATION


def test_HC03_misaligned_method_is_class_b_without_judging() -> None:
    e = ev(main=True, confirmed=True)
    u = classify(e, F_FOUND, None, MethodState.MISALIGNED)  # judge_h1 未被调用：没有结果传入
    assert u.result is B and u.b_reasons == {ClassBReason.METHOD_STATE_UNKNOWN}
    assert u.method_state is MethodState.MISALIGNED
    with pytest.raises(H1ContractError):  # 方法不完整却传入了 judge_h1 结果
        classify(e, F_FOUND, judged(H1Status.OK), MethodState.MISALIGNED)


def test_HC04_not_run_method_is_class_b() -> None:
    u = classify(ev(main=True, confirmed=True), F_FOUND, None, MethodState.NOT_RUN)
    assert u.result is B and u.b_reasons == {ClassBReason.METHOD_STATE_UNKNOWN}
    assert u.method_state is MethodState.NOT_RUN


def test_HC05_main_fref_absent_not_judged() -> None:
    u = classify(ev(main=True, confirmed=True), F_ABSENT)
    assert u.result is UnitResult.NOT_JUDGED


def test_HC06_delisted_prefix_complete_fref_found_judged() -> None:
    e = ev(prefix=True, censor=("RIGHT_DELISTED",))
    u = classify(e, F_FOUND, judged(H1Status.OK))
    assert u.result is UnitResult.OK  # 退市截尾照常判定，不进甲类


def test_HC07_delisted_prefix_complete_no_fref_is_class_a() -> None:
    u = classify(ev(prefix=True, censor=("RIGHT_DELISTED",)), F_ABSENT)
    assert u.result is A
    assert u.codes == {"RIGHT_DELISTED", "F_REF_ABSENT"} and u.categories == {DEL}


def test_HC08_period_end_prefix_no_fref_right_censored() -> None:
    u = classify(ev(prefix=True, censor=("RIGHT_PERIOD_END",)), F_ABSENT)
    assert u.result is A and u.categories == {RIGHT}


def test_HC09_missing_and_right_censor_is_one_class_a_event() -> None:
    e = ev(censor=("RIGHT_DATA_END", "MISSING"), unevaluable=("MISSING",))
    u = classify(e, unsearched(e))
    assert u.result is A
    assert u.codes == {"RIGHT_DATA_END", "MISSING", "UNEVALUABLE_MISSING"}
    assert u.categories == {MIS, RIGHT}
    s = summarize(
        [u],
        MethodKind.DAILY,
        (date(2010, 1, 1), date(2020, 1, 1)),
        DatasetRole.EXPLORATION,
        CAPS_NOT_APPROVED,
    )
    assert s.class_a_events == 1


def test_HC10_confirmed_missing_class_a() -> None:
    e = ev(confirmed=True, censor=("MISSING",))
    u = classify(e, unsearched(e))
    assert u.result is A and u.categories == {MIS}


def test_HC11_confirmed_warmup_class_a() -> None:
    e = ev(confirmed=True, unevaluable=("WARMUP",))
    u = classify(e, unsearched(e))
    assert u.result is A and u.categories == {WARM}


def test_HC12_data_break_class_a() -> None:
    e = ev(censor=("DATA_BREAK",))
    u = classify(e, unsearched(e))
    assert u.result is A and u.categories == {MIS}


def test_HC13_unknown_right_code_class_b() -> None:
    e = ev(censor=("RIGHT_FOO",))
    u = classify(e, unsearched(e))
    assert u.result is B and u.b_reasons == {ClassBReason.UNRECOGNIZED_REASON}
    assert u.unrecognized == {"RIGHT_FOO"}


def test_HC14_recognized_plus_unknown_is_class_b() -> None:
    e = ev(censor=("RIGHT_DATA_END", "XYZ"))
    u = classify(e, unsearched(e))
    assert u.result is B and u.b_reasons == {ClassBReason.UNRECOGNIZED_REASON}
    assert u.unrecognized == {"XYZ"}


def test_HC15_empty_reason_set_class_b() -> None:
    e = ev()
    u = classify(e, FRef(False, None, None, frozenset()))
    assert u.result is B and u.b_reasons == {ClassBReason.UNRECOGNIZED_REASON}
    assert u.unrecognized == frozenset()


def test_HC16_fref_absent_alone_unrecognized() -> None:
    c = classify_reasons(frozenset({"F_REF_ABSENT"}))
    assert not c.class_a and c.unrecognized == {"F_REF_ABSENT"} and not c.dependency_unverified


def test_HC16b_contract_searched_without_main_or_prefix() -> None:
    e = ev(censor=("RIGHT_DATA_END",), unevaluable=("MISSING",))
    with pytest.raises(H1ContractError):
        classify(e, F_ABSENT)


DEP_OK = (date(2015, 6, 1), DATA_END, 5)


def test_HC17_dependency_verified_event_and_interval_class_a() -> None:
    e = ev(prefix=True, e=date(2015, 6, 1), censor=("RIGHT_DATA_END",), dep=DEP_OK)
    u = classify(e, F_ABSENT)
    assert u.result is A
    assert u.codes == {"RIGHT_DATA_END", "F_REF_ABSENT", "DEPENDENCY_PENDING"}
    assert u.categories == {RIGHT}
    iv = interval(e)
    assert iv is not None and iv.kind is UnitKind.INTERVAL
    assert iv.result is A and iv.categories == {RIGHT} and iv.as_of == e.p


def test_HC18_dependency_delisted() -> None:
    e = ev(prefix=True, e=date(2015, 6, 1), censor=("RIGHT_DELISTED",), dep=DEP_OK)
    assert classify(e, F_ABSENT).categories == {DEL}
    iv = interval(e)
    assert iv is not None and iv.result is A and iv.categories == {DEL}


def test_HC19_dependency_start_mismatch_unverified() -> None:
    e = ev(
        prefix=True,
        e=date(2015, 6, 1),
        censor=("RIGHT_DATA_END",),
        dep=(date(2015, 5, 29), DATA_END, 5),
    )
    u = classify(e, F_ABSENT)
    assert u.result is B and u.b_reasons == {ClassBReason.DEPENDENCY_UNVERIFIED}
    iv = interval(e)
    assert (
        iv is not None and iv.result is B and iv.b_reasons == {ClassBReason.DEPENDENCY_UNVERIFIED}
    )


def test_HC20_empty_dependency_interval_no_interval_unit() -> None:
    e = ev(
        e=DATA_END,
        censor=("RIGHT_DATA_END",),
        unevaluable=("MISSING",),
        dep=(DATA_END, DATA_END, 0),
    )
    u = classify(e, unsearched(e))
    assert u.result is A
    assert u.codes == {"RIGHT_DATA_END", "UNEVALUABLE_MISSING", "DEPENDENCY_PENDING"}
    assert u.categories == {MIS, RIGHT}
    assert interval(e) is None
    assert expected_units(SEC, ESET, [e]) == (UnitId(SEC, ESET, e.s, e.p),)


def test_HC21_nonempty_interval_with_zero_valid_days() -> None:
    e = ev(
        prefix=True,
        e=date(2015, 6, 1),
        censor=("RIGHT_DATA_END",),
        dep=(date(2015, 6, 1), DATA_END, 0),
    )
    iv = interval(e)
    assert iv is not None and iv.result is A and iv.categories == {RIGHT} and iv.valid_days == 0


def test_HC22_violation_event_with_class_a_interval() -> None:
    e = ev(prefix=True, e=date(2015, 6, 1), censor=("RIGHT_DATA_END",), dep=DEP_OK)
    assert classify(e, F_FOUND, judged(H1Status.VIOLATION)).result is UnitResult.VIOLATION
    iv = interval(e)
    assert iv is not None and iv.result is A and iv.categories == {RIGHT}


def test_HC23_dependency_end_not_data_end_unverified() -> None:
    e = ev(
        prefix=True,
        e=date(2015, 6, 1),
        censor=("RIGHT_DATA_END",),
        dep=(date(2015, 6, 1), date(2019, 12, 30), 5),
    )
    assert classify(e, F_ABSENT).b_reasons == {ClassBReason.DEPENDENCY_UNVERIFIED}
    iv = interval(e)
    assert (
        iv is not None and iv.result is B and iv.b_reasons == {ClassBReason.DEPENDENCY_UNVERIFIED}
    )


def test_HC24_two_right_codes_unverified() -> None:
    e = ev(e=date(2015, 6, 1), censor=("RIGHT_DATA_END", "RIGHT_PERIOD_END"), dep=DEP_OK)
    u = classify(e, unsearched(e))
    assert u.result is B and u.b_reasons == {ClassBReason.DEPENDENCY_UNVERIFIED}
    iv = interval(e)
    assert (
        iv is not None and iv.result is B and iv.b_reasons == {ClassBReason.DEPENDENCY_UNVERIFIED}
    )


def test_HC25_unproven_warmup_class_b() -> None:
    e = ev(confirmed=True, unevaluable=("UNKNOWN",))
    u = classify(e, unsearched(e))
    assert u.result is B and u.b_reasons == {ClassBReason.UNRECOGNIZED_REASON}
    assert u.unrecognized == {"UNEVALUABLE_UNKNOWN"}


def test_HC26_dependency_unverified_and_unrecognized_both_kept() -> None:
    e = ev(
        e=date(2015, 6, 1),
        censor=("RIGHT_DATA_END", "XYZ"),
        dep=(date(2015, 5, 29), DATA_END, 5),
    )
    assert not e.is_prefix_complete
    u = classify(e, unsearched(e))
    assert u.result is B
    assert u.b_reasons == {ClassBReason.DEPENDENCY_UNVERIFIED, ClassBReason.UNRECOGNIZED_REASON}
    assert u.unrecognized == {"XYZ"}


def test_HC27_contract_fref_found_for_unsearchable_event() -> None:
    with pytest.raises(H1ContractError):
        classify(ev(censor=("RIGHT_DATA_END", "MISSING")), F_FOUND)


def test_HC28_contract_reasons_differ() -> None:
    e = ev(censor=("RIGHT_DATA_END", "MISSING"))
    assert exclusion_reasons(e) == {"RIGHT_DATA_END", "MISSING"}
    with pytest.raises(H1ContractError):
        classify(e, FRef(False, None, None, frozenset({"RIGHT_DATA_END"})))


def test_HC29_contract_main_must_be_searched() -> None:
    with pytest.raises(H1ContractError):
        classify(
            ev(main=True, confirmed=True), FRef(False, None, None, frozenset({"RIGHT_DELISTED"}))
        )


def test_HC30_contract_prefix_must_be_searched() -> None:
    with pytest.raises(H1ContractError):
        classify(
            ev(prefix=True, censor=("RIGHT_DATA_END",)),
            FRef(False, None, None, frozenset({"RIGHT_DATA_END"})),
        )


# ---- 2.3 方法完整性（HM 组） ----

CAL = (date(2015, 3, 2), date(2015, 3, 3), date(2015, 3, 4), date(2015, 3, 5))


def mseries(days: tuple[date, ...]) -> MethodSeries:
    n = len(days)
    return MethodSeries("R0", days, (None,) * n, (None,) * n, ())


def test_HM01_same_days_complete() -> None:
    assert method_state(CAL, mseries(CAL)) is MethodState.COMPLETE


def test_HM02_missing_last_day_misaligned() -> None:
    assert method_state(CAL, mseries(CAL[:-1])) is MethodState.MISALIGNED


def test_HM03_swapped_days_misaligned() -> None:
    swapped = (CAL[0], CAL[2], CAL[1], CAL[3])
    assert method_state(CAL, mseries(swapped)) is MethodState.MISALIGNED


def test_HM04_off_calendar_day_misaligned() -> None:
    other = (CAL[0], CAL[1], date(2015, 3, 7), CAL[3])
    assert method_state(CAL, mseries(other)) is MethodState.MISALIGNED


@pytest.mark.parametrize("given", [MethodState.NOT_RUN, MethodState.FAILED])
def test_HM05_no_series_recorded_as_given(given: MethodState) -> None:
    assert method_state(CAL, None, given) is given
    u = classify(ev(main=True, confirmed=True), F_FOUND, None, given)
    assert u.result is B and u.b_reasons == {ClassBReason.METHOD_STATE_UNKNOWN}
    with pytest.raises(H1ContractError):
        method_state(CAL, None, None)


# ---- 2.4 预热证明（HW 组） ----


def _classify_chain_event(series: EvalSeries, event: SevereEvent) -> H1Unit:
    return classify_h1(
        SEC,
        ESET,
        M,
        event,
        find_f_ref(series, event),
        None,
        MethodState.COMPLETE,
        series.calendar.last,
    )


def test_HW01_A31_first_event_warmup_proven() -> None:
    series, chain = run_case(load_fixtures()["A"]["A31"])
    first = chain.events[0]
    assert first.unevaluable == {"WARMUP"}
    u = _classify_chain_event(series, first)
    assert u.result is A and u.categories == {WARM}


def test_HW02_A39b_missing_and_warmup() -> None:
    series, chain = run_case(load_fixtures()["A"]["A39b"])
    (event,) = chain.events
    assert event.censor == {"MISSING"} and event.unevaluable == {"MISSING", "WARMUP"}
    u = _classify_chain_event(series, event)
    assert u.result is A
    assert u.codes == {"MISSING", "UNEVALUABLE_MISSING", "UNEVALUABLE_WARMUP"}
    assert u.categories == {MIS, WARM}


def _hw03_series() -> EvalSeries:
    seg2 = [100.0] * 70 + [100.0 - 2 * k for k in range(1, 31)]
    closes: list[float | None] = [50.0] * 120 + [None] * 11 + seg2
    days = workdays(len(closes))
    bars = [
        None if c is None else DailyBar(d, c, c + 1, c - 1, c, 1, c, c + 1, c - 1, c)
        for d, c in zip(days, closes, strict=True)
    ]
    return EvalSeries(TradingCalendar(days), tuple(bars), 10, 83, EndReason.DATA_END)


def test_HW03_new_segment_after_break_warmup() -> None:
    series = _hw03_series()
    seg2_days = [
        d for d, b in zip(series.calendar.days, series.bars, strict=True) if b is not None
    ][120:]
    spec = EventSetSpec(EventKind.PCT, Decimal("0.20"), "E_pct")
    chain = build_events(series, spec, EventParameters(Decimal("0.5"), 20))
    (event,) = chain.events  # 第 1 段不产生事件
    assert event.p == seg2_days[69] and event.s == seg2_days[82]  # 新段第 70、83 个有效日
    assert event.confirm is None and event.censor == {"RIGHT_DATA_END"}
    assert event.unevaluable == {"WARMUP"}
    u = _classify_chain_event(series, event)
    assert u.result is A
    assert u.codes == {"RIGHT_DATA_END", "UNEVALUABLE_WARMUP"} and u.categories == {RIGHT, WARM}


def test_HW04_count_sufficient_but_excluded_is_unknown() -> None:
    # 模拟今后新增的掩码排除原因；现行代码不会产生
    case = load_fixtures()["A"]["A31"]
    series = make_series(case)
    i = 168
    assert series.bars[i] is not None and series.segment_counts[i] >= 83 and series.mask[i]
    mask = list(series.mask)
    mask[i] = False
    object.__setattr__(series, "mask", tuple(mask))
    spec = EventSetSpec(EventKind.PCT, Decimal(case["event_set"]["T"]), case["id"])
    chain = build_events(series, spec, EventParameters(Decimal("0.5"), 20))
    days = series.calendar.days
    assert len(chain.events) == 2
    first, second = chain.events
    # 事件的生成（开始日、峰值、确认日）不变；变了就停止报告
    assert (second.s, second.p, second.confirm) == (days[167], days[163], days[171])
    assert first.unevaluable == {"WARMUP"}
    assert "UNKNOWN" in second.unevaluable and not second.is_main
    u = _classify_chain_event(series, second)
    assert "UNEVALUABLE_UNKNOWN" in u.codes
    assert u.result is B and u.b_reasons == {ClassBReason.UNRECOGNIZED_REASON}  # 不得归甲


def test_HW05_existing_event_fixtures_unchanged() -> None:
    path = Path(__file__).parent / "fixtures" / "eval_fixtures_v1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FIXTURE_SHA
    for case_id in load_fixtures()["A"]:
        assert_case(case_id)


# ---- 2.5 汇总（HA 组） ----

DEV = (date(2010, 1, 1), date(2020, 1, 1))
_seq = iter(range(1, 10_000))


def unit(
    result: UnitResult,
    kind: UnitKind = UnitKind.EVENT,
    cats: frozenset[Category] = frozenset(),
    p: date = date(2015, 6, 1),
    b: frozenset[ClassBReason] = frozenset(),
) -> H1Unit:
    n = next(_seq)
    s = date(2015, 1, 1).fromordinal(date(2015, 1, 1).toordinal() + n)
    uid = (
        UnitId(SEC, ESET, s, p)
        if kind is UnitKind.EVENT
        else UnitId(SEC, ESET, s, p, date(2016, 1, 1), DATA_END)
    )
    return H1Unit(
        uid,
        kind,
        M,
        result,
        frozenset(),
        cats,
        b,
        frozenset(),
        MethodState.COMPLETE,
        p,
        3 if kind is UnitKind.INTERVAL else None,
    )


def run(units, caps=CAPS_NOT_APPROVED, role=DatasetRole.EXPLORATION, period=DEV):  # noqa: ANN001, ANN201
    return summarize(units, MethodKind.DAILY, period, role, caps)


def ha01_units() -> list[H1Unit]:
    return [
        unit(UnitResult.OK),
        unit(UnitResult.OK),
        unit(UnitResult.OK),
        unit(UnitResult.VIOLATION),
        unit(B, b=frozenset({ClassBReason.METHOD_STATE_UNKNOWN})),
        unit(A, cats=frozenset({DEL})),
        unit(A, cats=frozenset({MIS, RIGHT})),
        unit(A, UnitKind.INTERVAL, frozenset({RIGHT})),
        unit(UnitResult.NOT_JUDGED),
    ]


def q(r) -> Decimal | None:  # noqa: ANN001
    return r.q12


def test_HA01_counts_ratios_and_sensitivity() -> None:
    s = run(ha01_units())
    assert (s.checkable_events, s.checkable_intervals, s.checkable) == (7, 1, 8)
    assert s.not_judged == 1
    assert (s.class_a_events, s.class_a_events + s.class_a_intervals) == (2, 3)
    assert s.event_categories == {DEL: 1, MIS: 1, RIGHT: 1}
    assert s.interval_categories == {RIGHT: 1}
    assert s.violations_for_eligibility == 2 and s.sensitivity_violations == 5
    assert s.class_b == {ClassBReason.METHOD_STATE_UNKNOWN: 1}
    assert (s.event_class_a.numerator, s.event_class_a.denominator) == (2, 7)
    assert q(s.event_class_a) == Decimal("0.285714285714")
    assert (s.mixed_class_a.numerator, s.mixed_class_a.denominator) == (3, 8)
    assert q(s.mixed_class_a) == Decimal("0.375000000000")
    assert (s.event_delisted.numerator, s.event_delisted.denominator) == (1, 7)
    assert q(s.event_delisted) == Decimal("0.142857142857")
    assert (s.mixed_delisted.numerator, s.mixed_delisted.denominator) == (1, 8)
    assert q(s.mixed_delisted) == Decimal("0.125000000000")
    assert s.cap_conclusion is CapConclusion.NOT_APPROVED_DISPLAY_ONLY
    assert s.cap_conclusion.value == "上限未批准，只展示"


def test_HA02_test_caps_equal_passes_but_delisted_exceeds() -> None:
    s = run(ha01_units(), Caps(Decimal("0.375"), Decimal("0.125")))  # 测试值，不是拟定值
    assert s.cap_conclusion is CapConclusion.INSUFFICIENT_EVIDENCE and s.exceeded == ("③",)


def test_HA03_within_caps_but_h1_still_fails() -> None:
    s = run(ha01_units(), Caps(Decimal("0.375"), Decimal("0.15")))  # 测试值，不是拟定值
    assert s.cap_conclusion is CapConclusion.WITHIN
    assert s.violations_for_eligibility == 2  # 上限内不改变违反数，H1 不通过


def test_HA04_zero_event_denominator_undeterminable() -> None:
    units = [unit(UnitResult.NOT_JUDGED), unit(A, UnitKind.INTERVAL, frozenset({RIGHT}))]
    s = run(units)
    assert not s.event_class_a.determinable and not s.event_delisted.determinable
    assert s.event_class_a.q12 is None
    assert q(s.mixed_class_a) == Decimal("1.000000000000")
    assert q(s.mixed_delisted) == Decimal("0.000000000000")
    s2 = run(units, Caps(Decimal("0.375"), Decimal("0.125")))  # 测试值
    assert s2.cap_conclusion is CapConclusion.UNDETERMINABLE


def test_HA05_empty_expected_list() -> None:
    assert expected_units(SEC, ESET, []) == ()
    s = run([])
    assert (s.checkable, s.not_judged, s.violations_for_eligibility) == (0, 0, 0)
    for r in (s.event_class_a, s.mixed_class_a, s.event_delisted, s.mixed_delisted):
        assert not r.determinable


def test_HA06_period_by_attribution_date() -> None:
    units = [
        unit(UnitResult.OK, p=date(2009, 12, 31)),
        unit(UnitResult.OK, p=date(2019, 12, 31)),
        unit(A, UnitKind.INTERVAL, frozenset({RIGHT}), p=date(2019, 6, 3)),
        unit(A, UnitKind.INTERVAL, frozenset({RIGHT}), p=date(2009, 11, 2)),
    ]
    s = run(units)
    assert (s.checkable_events, s.checkable_intervals, s.checkable) == (1, 1, 2)


@pytest.mark.parametrize("role", [DatasetRole.SELECTION, DatasetRole.VALIDATION])
def test_HA07_view_restriction(role: DatasetRole) -> None:
    with pytest.raises(H1ContractError):
        run(ha01_units(), CAPS_NOT_APPROVED, role)
    assert run(ha01_units(), CAPS_NOT_APPROVED, DatasetRole.EXPLORATION).checkable == 8


def _events(n: int, class_a: int) -> list[H1Unit]:
    return [unit(A, cats=frozenset({MIS})) for _ in range(class_a)] + [
        unit(UnitResult.OK) for _ in range(n - class_a)
    ]


def test_HA08_exact_comparison_one_third() -> None:
    s = run(_events(3, 1), Caps(Decimal("0.333333333333"), Decimal(1)))  # 测试值
    assert q(s.event_class_a) == Decimal("0.333333333333")
    assert s.cap_conclusion is CapConclusion.INSUFFICIENT_EVIDENCE


def test_HA09_equal_then_one_more_exceeds() -> None:
    caps = Caps(Decimal("0.25"), Decimal(1))  # 测试值
    assert run(_events(8, 2), caps).cap_conclusion is CapConclusion.WITHIN
    assert run(_events(9, 3), caps).cap_conclusion is CapConclusion.INSUFFICIENT_EVIDENCE


def test_HA10_zero_caps() -> None:
    caps = Caps(Decimal(0), Decimal(0))  # 测试值
    assert run(_events(3, 0), caps).cap_conclusion is CapConclusion.WITHIN
    assert run(_events(3, 1), caps).cap_conclusion is CapConclusion.INSUFFICIENT_EVIDENCE
    assert run([unit(UnitResult.NOT_JUDGED)], caps).cap_conclusion is CapConclusion.UNDETERMINABLE


@pytest.mark.parametrize(
    ("u_a", "u_d"),
    [
        (0.25, Decimal(1)),
        (Decimal("1.0000001"), Decimal(1)),
        (Decimal("0.5"), Decimal("-0.01")),
        (Decimal("NaN"), Decimal(1)),
        (Decimal("Infinity"), Decimal(1)),
        (Decimal("0.5"), Decimal("-Infinity")),
        (True, Decimal(1)),
        (0, Decimal(1)),
        (1, Decimal(1)),
    ],
)
def test_HA11_cap_input_validation(u_a: object, u_d: object) -> None:
    with pytest.raises(H1ContractError):
        Caps(u_a, u_d)  # type: ignore[arg-type]


def test_HA12_reverse_quantization_boundary() -> None:
    s = run(_events(3, 2), Caps(Decimal("0.6666666666667"), Decimal(1)))  # 测试值
    assert q(s.event_class_a) == Decimal("0.666666666667")
    assert s.cap_conclusion is CapConclusion.WITHIN


# ---- 2.6 覆盖与跨方法一致（HX 组） ----

NAMES = ("R0", "B1", "B2", "B3", "C1", "C2")
E_MAIN = ev(main=True, confirmed=True, s=date(2015, 3, 5), p=date(2015, 3, 2))
E_A = ev(confirmed=True, censor=("MISSING",), s=date(2015, 7, 5), p=date(2015, 7, 2))
E_DEP = ev(
    prefix=True,
    e=date(2015, 6, 1),
    censor=("RIGHT_DATA_END",),
    dep=DEP_OK,
    s=date(2015, 5, 5),
    p=date(2015, 5, 1),
)
CHAIN = (E_MAIN, E_A, E_DEP)


def items(state: MethodState | None) -> list[EventInput]:
    refs = {E_MAIN: F_FOUND, E_A: unsearched(E_A), E_DEP: F_ABSENT}
    out = []
    for e in CHAIN:
        j = judged(H1Status.OK) if (state is MethodState.COMPLETE and e is E_MAIN) else None
        out.append(EventInput(e, refs[e], j))
    return out


def all_results(c2_state: MethodState = MethodState.COMPLETE) -> dict[str, tuple[H1Unit, ...]]:
    out = {}
    for name in NAMES:
        if name == "B3":
            out[name] = method_units(
                SEC, ESET, name, MethodKind.NO_DAILY_STATE, None, items(None), DATA_END
            )
        else:
            state = c2_state if name == "C2" else MethodState.COMPLETE
            out[name] = method_units(
                SEC, ESET, name, MethodKind.DAILY, state, items(state), DATA_END
            )
    return out


EXPECTED = expected_units(SEC, ESET, CHAIN)


def test_HX01_all_methods_pass_and_b3_not_applicable() -> None:
    results = all_results()
    common = check_units(EXPECTED, NAMES, results)
    assert len(EXPECTED) == 4 and len(common) == 3  # 甲：E_A、E_DEP 事件与它的区间
    b3 = {u.unit_id: u for u in results["B3"]}
    assert (
        b3[UnitId(SEC, ESET, E_MAIN.s, E_MAIN.p)].result is UnitResult.NOT_APPLICABLE_NO_DAILY_STATE
    )
    with pytest.raises(H1ContractError):  # B3 不计入任何违反数：汇总直接拒绝
        summarize(
            results["B3"],
            MethodKind.NO_DAILY_STATE,
            DEV,
            DatasetRole.EXPLORATION,
            CAPS_NOT_APPROVED,
        )


def test_HX02_missing_unit_raises() -> None:
    results = all_results()
    results["C2"] = results["C2"][1:]
    with pytest.raises(H1ContractError, match="漏"):
        check_units(EXPECTED, NAMES, results)


def test_HX03_duplicate_unit_raises() -> None:
    results = all_results()
    results["C2"] = (*results["C2"], results["C2"][0])
    with pytest.raises(H1ContractError, match="重复"):
        check_units(EXPECTED, NAMES, results)


def test_HX04_tampered_category_raises() -> None:
    results = all_results()
    units = list(results["C2"])
    k = next(i for i, u in enumerate(units) if u.result is A)
    units[k] = replace(units[k], categories=frozenset({DEL}))
    results["C2"] = tuple(units)
    with pytest.raises(H1ContractError, match="甲类集合"):
        check_units(EXPECTED, NAMES, results)


def test_HX05_b3_class_a_set_compared_too() -> None:
    results = all_results()
    units = list(results["B3"])
    k = next(i for i, u in enumerate(units) if u.result is A)
    units[k] = replace(
        units[k], result=UnitResult.NOT_APPLICABLE_NO_DAILY_STATE, categories=frozenset()
    )
    results["B3"] = tuple(units)
    with pytest.raises(H1ContractError, match="甲类集合"):
        check_units(EXPECTED, NAMES, results)


def test_HX06_empty_expected_and_results() -> None:
    expected = expected_units(SEC, ESET, [])
    assert expected == ()
    common = check_units(expected, NAMES, {n: () for n in NAMES})
    assert common == frozenset()


def test_HX07_whole_method_missing_raises() -> None:
    results = all_results()
    del results["C2"]
    with pytest.raises(H1ContractError, match="缺"):
        check_units(EXPECTED, NAMES, results)


def test_HX08_extra_method_raises() -> None:
    results = all_results()
    results["C3"] = tuple(replace(u, method="C3") for u in results["C1"])
    with pytest.raises(H1ContractError, match="多"):
        check_units(EXPECTED, NAMES, results)


def test_HX09_unit_outside_expected_list_raises() -> None:
    results = all_results()
    extra = replace(
        results["C1"][0], unit_id=UnitId(SEC, ESET, date(2015, 9, 1), date(2015, 8, 31))
    )
    results["C1"] = (*results["C1"], extra)
    with pytest.raises(H1ContractError, match="清单外"):
        check_units(EXPECTED, NAMES, results)


def test_HX10_not_run_method_still_checked() -> None:
    results = all_results(MethodState.NOT_RUN)
    common = check_units(EXPECTED, NAMES, results)
    c2 = {u.unit_id: u for u in results["C2"]}
    main_unit = c2[UnitId(SEC, ESET, E_MAIN.s, E_MAIN.p)]
    assert main_unit.result is B and main_unit.b_reasons == {ClassBReason.METHOD_STATE_UNKNOWN}
    s = summarize(results["C2"], MethodKind.DAILY, DEV, DatasetRole.EXPLORATION, CAPS_NOT_APPROVED)
    assert s.violations_for_eligibility == 1
    from stock_risk.domain.evaluation.h1_classes import class_a_set

    assert class_a_set(results["C2"]) == common == class_a_set(results["R0"])


def test_HX11_b3_rejected_by_summary() -> None:
    results = all_results()
    with pytest.raises(H1ContractError):
        summarize(
            results["B3"],
            MethodKind.NO_DAILY_STATE,
            DEV,
            DatasetRole.EXPLORATION,
            CAPS_NOT_APPROVED,
        )
    with pytest.raises(H1ContractError):  # 即使种类写成逐日，出现不适用单位也拒绝
        summarize(results["B3"], MethodKind.DAILY, DEV, DatasetRole.EXPLORATION, CAPS_NOT_APPROVED)


def test_HX12_b3_keeps_source_reasons() -> None:
    e_xyz = ev(
        e=date(2015, 6, 1),
        censor=("RIGHT_DATA_END", "XYZ"),
        dep=(date(2015, 5, 29), DATA_END, 5),
        s=date(2015, 5, 5),
        p=date(2015, 5, 1),
    )
    chain = (E_A, e_xyz)
    expected = expected_units(SEC, ESET, chain)
    assert len(expected) == 3
    its = [EventInput(E_A, unsearched(E_A)), EventInput(e_xyz, unsearched(e_xyz))]
    b3 = method_units(SEC, ESET, "B3", MethodKind.NO_DAILY_STATE, None, its, DATA_END)
    r0 = method_units(SEC, ESET, "R0", MethodKind.DAILY, MethodState.COMPLETE, its, DATA_END)
    by_b3 = {u.unit_id: u for u in b3}
    by_r0 = {u.unit_id: u for u in r0}
    a_id = UnitId(SEC, ESET, E_A.s, E_A.p)
    xyz_id = UnitId(SEC, ESET, e_xyz.s, e_xyz.p)
    iv_id = next(i for i in expected if i.kind is UnitKind.INTERVAL)
    assert by_b3[a_id].result is A and by_b3[a_id].categories == {MIS}
    na = UnitResult.NOT_APPLICABLE_NO_DAILY_STATE
    assert by_b3[xyz_id].result is na and by_b3[xyz_id].unrecognized == {"XYZ"}
    assert by_b3[iv_id].result is na and by_b3[iv_id].b_reasons == {
        ClassBReason.DEPENDENCY_UNVERIFIED
    }
    assert by_r0[xyz_id].result is B and by_r0[iv_id].result is B
    check_units(expected, ("R0", "B3"), {"R0": r0, "B3": b3})


# ---- 2.7 端到端（现有夹具，不改 JSON） ----


def _daily_units(case_id: str, group: str = "B"):  # noqa: ANN202
    case = b_case(case_id) if group == "B" else load_fixtures()["A"][case_id]
    series, chain = run_case(case)
    refs = [find_f_ref(series, e) for e in chain.events]
    return case, series, chain, refs


def test_E2E_B14_same_as_existing_results() -> None:
    case, series, chain, refs = _daily_units("B14")
    got = []
    for name, m in methods(series, case).items():
        state = method_state(series.calendar.days, m)
        assert state is MethodState.COMPLETE
        (event,) = chain.events
        u = classify_h1(
            SEC,
            "B14",
            name,
            event,
            refs[0],
            judge_h1(series, m, event, refs[0]),
            state,
            series.calendar.last,
        )
        got.append(u.result.value)
    assert got == ["OK", "VIOLATION", "VIOLATION", "OK"]


def test_E2E_B14n_not_judged_zero_denominator() -> None:
    case, series, chain, refs = _daily_units("B14n")
    ((name, m),) = methods(series, case).items()
    (event,) = chain.events
    u = classify_h1(
        SEC, "B14n", name, event, refs[0], None, MethodState.COMPLETE, series.calendar.last
    )
    assert u.result is UnitResult.NOT_JUDGED
    s = summarize(
        [u],
        MethodKind.DAILY,
        (date(2010, 1, 1), date(2020, 1, 1)),
        DatasetRole.EXPLORATION,
        CAPS_NOT_APPROVED,
    )
    assert s.checkable == 0
    for r in (s.event_class_a, s.mixed_class_a, s.event_delisted, s.mixed_delisted):
        assert not r.determinable


def test_E2E_B21_judged_as_usual() -> None:
    case, series, chain, refs = _daily_units("B21")
    (event,) = chain.events
    got = {
        name: classify_h1(
            SEC,
            "B21",
            name,
            event,
            refs[0],
            judge_h1(series, m, event, refs[0]),
            MethodState.COMPLETE,
            series.calendar.last,
        ).result.value
        for name, m in methods(series, case).items()
    }
    assert got == {"R_ALERT_FROM_95": "OK", "R_NO_ALERT": "VIOLATION"}


def test_E2E_B21m_both_methods_class_a() -> None:
    case, series, chain, refs = _daily_units("B21m")
    names = tuple(methods(series, case))
    results = {
        n: method_units(
            SEC,
            "B21m",
            n,
            MethodKind.DAILY,
            MethodState.COMPLETE,
            [EventInput(chain.events[0], refs[0])],
            series.calendar.last,
        )
        for n in names
    }
    for units in results.values():
        (u,) = units
        assert u.result is A
        assert u.codes == {"MISSING", "RIGHT_DATA_END", "UNEVALUABLE_MISSING"}
        assert u.categories == {MIS, RIGHT}
    check_units(expected_units(SEC, "B21m", chain.events), names, results)


def test_E2E_A_M3_dependency_interval() -> None:
    case, series, chain, refs = _daily_units("A_M3", "A")
    (event,) = chain.events
    days = series.calendar.days
    never = make_method(series, "QUIET", {"spans": [], "events": []})
    alert = make_method(
        series, "ALERTED", {"spans": [[100, 110, "ep1", "WATCH", {}]], "events": []}
    )
    results = {}
    for m in (never, alert):
        state = method_state(days, m)
        assert state is MethodState.COMPLETE
        results[m.name] = method_units(
            SEC,
            "A_M3",
            m.name,
            MethodKind.DAILY,
            state,
            [EventInput(event, refs[0])],
            series.calendar.last,
        )
    for units in results.values():
        ev_u = next(u for u in units if u.kind is UnitKind.EVENT)
        iv_u = next(u for u in units if u.kind is UnitKind.INTERVAL)
        assert len(units) == 2
        assert ev_u.result is A
        assert ev_u.codes == {"DEPENDENCY_PENDING", "MISSING", "RIGHT_DATA_END"}
        assert ev_u.categories == {MIS, RIGHT}
        assert iv_u.result is A and iv_u.categories == {RIGHT}
        assert iv_u.valid_days == 6 and iv_u.as_of == days[99]
    check_units(expected_units(SEC, "A_M3", chain.events), ("QUIET", "ALERTED"), results)
