"""T09-E4：汇总与选参函数的单元测试（期望值按 17 号定义手算）。"""

from dataclasses import fields, replace
from decimal import Decimal

import pytest

from stock_risk.domain.evaluation.aggregate import (
    ALL,
    GATES,
    GateCell,
    MinimumSamples,
    Period,
    SampleStatus,
    Thresholds,
    adverse_rate,
    calibration_count,
    concentration,
    h_tail_grouping,
    judge_all,
    s_share,
    share,
    split_segments,
    unalerted_rate,
)
from stock_risk.domain.evaluation.membership import Segment
from stock_risk.domain.evaluation.outcomes import OutcomeClass
from stock_risk.domain.evaluation.selection import (
    EligibilityInput,
    eligibility,
    formal_violations,
    neighbor_points,
    rank,
)
from stock_risk.domain.lifecycle.params import (
    GRID_DIMENSIONS,
    EngineParams,
    ExplorationAssumptions,
    LongPath,
    ReleaseHyst,
    V08Switches,
    candidate_grid,
)
from stock_risk.domain.numeric import InvalidInputError
from tests.unit.evaluation.helpers import load_fixtures, run_case

D = Decimal
GATE_VALUES = dict.fromkeys(GATES, D("0.5"))


def test_thresholds_and_minimums_have_no_defaults() -> None:
    """门槛与最小样本只由调用方传入：两个数据类的每个字段都没有默认值。"""
    for cls in (Thresholds, MinimumSamples):
        for f in fields(cls):
            assert f.default is f.default_factory, (cls.__name__, f.name)  # 都是 MISSING
        with pytest.raises(TypeError):
            cls()  # type: ignore[call-arg]
    with pytest.raises(InvalidInputError):
        Thresholds(**{**GATE_VALUES, "p": 0.1})  # type: ignore[arg-type]


def test_rates_empty_denominator_is_none() -> None:
    assert share(0, 0) is None
    assert unalerted_rate([OutcomeClass.EXCLUDED]) is None
    cls = [OutcomeClass.C, OutcomeClass.U, OutcomeClass.N, OutcomeClass.EXCLUDED]
    assert unalerted_rate(cls) == D("0.333333333333")  # 1 / 3，EXCLUDED 不计
    assert s_share([D(1), D("0.5"), D(1), D(0)]) == D("0.5")
    assert adverse_rate([True, False, False, False]) == D("0.25")


def test_concentration_ceiling_and_ties() -> None:
    c = concentration({f"S{i:02d}": 1 for i in range(1, 12)})  # 11 只：⌈1.1⌉ = 2
    assert c.top_n == 2 and c.top == ("S01", "S02") and c.share == D("0.181818181818")
    assert concentration({"A": 0, "B": 0}).share is None


def test_judge_all_states() -> None:
    gates = Thresholds(**GATE_VALUES)
    cells = [
        GateCell("m", ALL, D("0.5"), SampleStatus.OK, "E_pct", "MAIN"),  # 恰等于门槛：通过
        GateCell("m", "LOW", None, SampleStatus.NOT_APPLICABLE, "E_pct", "MAIN"),
    ]
    s = judge_all(cells, gates)
    assert s.all_pass and s.evidence_ok and s.results[0].passed is True
    s = judge_all([*cells, GateCell("p", ALL, D("0.6"), SampleStatus.OK)], gates)
    assert not s.all_pass and s.evidence_ok and len(s.failed) == 1
    s = judge_all([*cells, GateCell("p", ALL, None, SampleStatus.CANDIDATE_INSUFFICIENT)], gates)
    assert not s.all_pass and not s.evidence_ok and s.results[-1].passed is None
    with pytest.raises(InvalidInputError):
        GateCell("q", ALL, None, SampleStatus.OK)


def test_validation_h_tail_ungrouped_is_judged_not_undetermined() -> None:
    """T09-C1 E4-01 裁定：验证期 h_tail 未分组照常只按全体判定，不返回 UNDETERMINED。"""
    m = MinimumSamples(100, 30, 50, 15, 50, 15, 20, 50, 15, 20)
    h = h_tail_grouping({"LOW": 25, "MID": 19, "HIGH": 30}, 74, m, Period.VALIDATION)
    assert not h.grouped and h.overall is SampleStatus.OK


def test_h_tail_grouped_when_every_group_has_enough() -> None:
    m = MinimumSamples(100, 30, 50, 15, 50, 15, 20, 50, 15, 20)
    h = h_tail_grouping({"LOW": 20, "MID": 20, "HIGH": 21}, 61, m, Period.DEVELOPMENT)
    assert h.grouped and h.overall is SampleStatus.OK


def test_calibration_count_confirmed_uncensored() -> None:
    """A01（以 (c) 结束的上限事件计入）与 A30（待确认时数据终止：右端截尾，不计）。"""
    a = load_fixtures()["A"]
    _, a01 = run_case(a["A01"])
    _, a30 = run_case(a["A30"])
    assert calibration_count(a01.events) == sum(
        1 for e in a01.events if e.confirm is not None and not e.censor
    )
    assert calibration_count(a01.events) >= 1
    assert calibration_count([e for e in a30.events if e.censor]) == 0


def test_split_segments() -> None:
    s = split_segments([(Segment.FRONT, 1), (Segment.NEITHER, 2), (Segment.BACK, 3)])
    assert (s.front, s.back, s.neither) == ((1,), (3,), 1)


def test_neighbor_points_on_grid() -> None:
    grid = {name: values for name, values in GRID_DIMENSIONS}
    grid |= {"x1": (0.0, 0.08, 0.10), "h": (0.5,)}
    _, c = candidate_grid()[0]
    point = {name: getattr(c, name) for name, _ in GRID_DIMENSIONS} | {"x1": 0.08, "h": 0.5}
    got = neighbor_points(point, grid)
    changed = [next(k for k in p if p[k] != point[k]) for p in got]
    # n、k1、k2、k、d、u_rel 在网格边界各 1 个；x1 在中间 2 个；h 只有一个取值，没有相邻值
    assert changed == ["n", "k1", "k2", "k", "d", "u_rel", "x1", "x1"]
    assert all(p["confirm"] == point["confirm"] for p in got)  # 结构维度不变
    with pytest.raises(InvalidInputError):
        neighbor_points({**point, "n": 30}, grid)


def test_formal_violations_follow_engine_params() -> None:
    """组合约束以 EngineParams.is_formal_candidate 为准（17 §5.2 第 1 项）。"""
    _, c = candidate_grid()[0]
    v3 = V08Switches(
        x1=0.08, x2=0.15, t11_full=True, long_path=LongPath.X2, release_hyst=ReleaseHyst.EFFECTIVE
    )
    for sw, h in ((v3, 0.5), (replace(v3, release_hyst=ReleaseHyst.OLD), 0.5), (v3, 1.0)):
        p = EngineParams(c, ExplorationAssumptions(h=h), switches=sw)
        assert (formal_violations(p) == ()) is p.is_formal_candidate


def test_eligibility_reasons() -> None:
    ok = EligibilityInput(True, 0, True, True, True)
    assert eligibility(ok).eligible
    bad = EligibilityInput(False, 2, True, False, True)
    assert eligibility(bad).reasons == ("FORMAL", "H1", "EVIDENCE")


def test_rank_empty_set() -> None:
    r = rank([], D("0.01"), D("0.02"))
    assert r.selected is None and r.a1 == () and r.s1_star is None
