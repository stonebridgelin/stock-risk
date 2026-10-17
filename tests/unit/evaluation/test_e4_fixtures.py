"""T09-E4：18 号 D16、D18、D25、D43、D43r。

期望值逐项取自夹具 JSON；门槛、容差、最小样本取夹具值。
"""

from decimal import Decimal
from typing import Any

import pytest

from stock_risk.domain.evaluation.aggregate import (
    ALL,
    CalibrationStatus,
    GateCell,
    MinimumSamples,
    Period,
    SampleStatus,
    Thresholds,
    calibrate_x,
    concentration,
    h_tail_grouping,
    judge,
    middle,
    sample_status,
    tail,
)
from stock_risk.domain.evaluation.selection import NeighborResult, RankInput, rank, robustness
from tests.unit.evaluation.helpers import load_fixtures


def d_case(case_id: str) -> dict[str, Any]:
    return load_fixtures()["D"][case_id]


def dec(value: str) -> Decimal:
    return Decimal(value)


def thresholds(**given: str) -> Thresholds:
    """只填夹具给出的门槛；其余门槛与本夹具无关，取一个不会被用到的值（测试内显式写出）。"""
    unused = Decimal("-1")
    names = ("m", "p", "p_L", "s", "o_max", "o_tail", "h_tail", "F")
    return Thresholds(**{n: dec(given[n]) if n in given else unused for n in names})


# ---- D16 ----


def test_D16_concentrated_occupancy_o_tail_blocks() -> None:
    case = d_case("D16")
    occ = [dec(s["occupancy"]) for s in case["securities"]]
    exp = case["expected"]
    o_median, o_p90 = middle(occ), tail(occ)
    assert o_median == dec(exp["o_median"]) and o_p90 == dec(exp["o_p90"])
    gates = thresholds(**case["thresholds"])
    results = {
        g: judge(GateCell(g, ALL, v, SampleStatus.OK), gates)
        for g, v in (("o_max", o_median), ("o_tail", o_p90))
    }
    assert {g: "PASS" if r.passed else "FAIL" for g, r in results.items()} == {
        "o_max": exp["o_max"],
        "o_tail": exp["o_tail"],
    }
    c = concentration({s["id"]: s["C_events"] for s in case["securities"]})
    assert c.top_n == exp["concentration_top_n"] and c.top == ("S10",)
    assert c.share == dec(exp["concentration_share"])


# ---- D18 ----


def _minimums() -> MinimumSamples:
    return MinimumSamples(**d_case("D18")["minimums"])


def _cases() -> dict[str, str]:
    return {c["case"]: c["expected"].split("（")[0] for c in d_case("D18")["cases"]}


def test_D18_minimum_samples_development() -> None:
    m, expected = _minimums(), _cases()
    dev = Period.DEVELOPMENT
    # E_pct 主口径事件全体 99 个：由样本决定的分母
    assert sample_status(99, m.events_overall, dev) == expected["E_pct 主口径事件全体 99 个"]
    # 探索集没有高波动组的证券：结构上不可能
    got = sample_status(0, m.group_securities, dev, applicable=False)
    assert got == expected["探索集没有高波动组的证券（结构上不可能）"]
    # 候选 X：Q 窗口完整 49 个、Q_L 窗口完整 60 个，两个分母分别检查
    p = sample_status(49, m.releases, dev, candidate_denominator=True)
    p_l = sample_status(60, m.releases, dev, candidate_denominator=True)
    case = "候选 X：Q 窗口完整的 RELEASE 全体 49 个，Q_L 窗口完整的 60 个"
    assert expected[case].startswith("候选 X 的 p 为 CANDIDATE_INSUFFICIENT")
    assert p is SampleStatus.CANDIDATE_INSUFFICIENT and p_l is SampleStatus.OK
    # h_tail：低 25、中 19、高 30，全体 74 → 未分组，只按全体判定，标明各组窗口数
    h = h_tail_grouping({"LOW": 25, "MID": 19, "HIGH": 30}, 74, m, dev)
    assert expected["h_tail 主口径窗口：低 25、中 19、高 30，全体 74"].startswith("h_tail 未分组")
    assert not h.grouped and h.overall is SampleStatus.OK
    assert dict(h.group_counts) == {"LOW": 25, "MID": 19, "HIGH": 30}
    # h_tail 全体 15
    h15 = h_tail_grouping({"LOW": 5, "MID": 5, "HIGH": 5}, 15, m, dev)
    assert h15.overall == expected["h_tail 主口径窗口全体 15"]
    # 某波动组已归组证券 19 只
    assert sample_status(19, m.group_securities, dev) == expected["某波动组已归组证券 19 只"]


def test_D18_validation_period_undetermined() -> None:
    """验证期：17 §10A 的三种情形（样本分母不足、结构上不可能、候选分母不足）都记 UNDETERMINED。
    h_tail 未分组不在此列，照常只按全体判定（T09-C1 E4-01 裁定）。"""
    m, expected = _minimums(), _cases()
    val = Period.VALIDATION
    assert expected["验证期出现以上任一情形"] == "UNDETERMINED"
    for got in (
        sample_status(99, m.events_overall, val),
        sample_status(0, m.group_securities, val, applicable=False),
        sample_status(49, m.releases, val, candidate_denominator=True),
        h_tail_grouping({"LOW": 5, "MID": 5, "HIGH": 5}, 15, m, val).overall,
        sample_status(19, m.group_securities, val),
    ):
        assert got is SampleStatus.UNDETERMINED
    assert sample_status(60, m.releases, val, candidate_denominator=True) is SampleStatus.OK


# ---- D25 ----


@pytest.mark.parametrize("i", range(4))
def test_D25_x_calibration(i: int) -> None:
    case = d_case("D25")["cases"][i]
    got = calibrate_x(case["E_pct_events"], {dec(x): n for x, n in case["E_atr_events"].items()})
    if case["expected_X"] == "STOP_INSUFFICIENT":
        assert got.status is CalibrationStatus.STOP_INSUFFICIENT and got.x is None
    else:
        assert got.status is CalibrationStatus.OK and got.x == Decimal(case["expected_X"])
        zero = [dec(x) for x, n in case["E_atr_events"].items() if n == 0]
        assert not set(zero) & set(got.ratios)  # 事件数为 0 的 X 不参与


# ---- D43 ----


def test_D43_layered_ranking_boundaries() -> None:
    case = d_case("D43")
    inputs = [
        RankInput(
            c["grid"],
            dec(c["S1_pct"]),
            dec(c["S1_atr"]),
            dec(c["S2_pct"]),
            dec(c["S2_atr"]),
            dec(c["F"]),
        )
        for c in case["candidates"]
    ]
    r = rank(inputs, dec(case["eps1"]), dec(case["eps2"]))
    exp = case["expected"]
    assert {str(g): v for g, v in r.s1_mean.items()} == {
        k: dec(v) for k, v in exp["S1_mean"].items()
    }
    assert {str(g): v for g, v in r.s2_mean.items()} == {
        k: dec(v) for k, v in exp["S2_mean"].items()
    }
    assert r.s1_star == dec(exp["S1_star"]) and list(r.a1) == exp["A1"]
    assert r.s2_star == dec(exp["S2_star"]) and list(r.a2) == exp["A2"]
    assert r.selected == exp["selected"]


# ---- D43r ----


@pytest.mark.parametrize("i", range(3))
def test_D43r_robustness(i: int) -> None:
    case = d_case("D43r")
    by_id = {
        n["id"]: NeighborResult(
            n["id"],
            n["formal"],
            n["gates"] == "PASS",
            n["evidence"] == "OK",
            dec(n["S1"]),
            dec(n["S2"]),
        )
        for n in case["neighbors"]
    }
    exp = case["expected"][i]
    got = robustness(
        dec(case["self"]["S1"]),
        dec(case["self"]["S2"]),
        [by_id[n] for n in exp["neighbors"]],
        dec(case["delta1"]),
        dec(case["delta2"]),
    )
    assert got.robust is exp["robust"], exp["note"]
    if "n4" in exp["neighbors"]:
        assert got.skipped == ("n4",)
