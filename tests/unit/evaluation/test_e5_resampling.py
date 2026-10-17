"""T09-E5：18 号 E17（逐次的 w、v、方法均值、Δ*，有效重复数与 p；L_b = 1 与 2），
以及 Holm 校正、B_v 阈值与块划分的单元测试（期望按定义手算）。"""

import random
from datetime import date, timedelta
from decimal import Decimal

import pytest

from stock_risk.domain.evaluation.resampling import (
    CalendarBlocks,
    Observation,
    Replicate,
    bootstrap,
    draw,
    holm,
    improvement_test,
)
from stock_risk.domain.numeric import InvalidInputError
from tests.unit.evaluation.helpers import load_fixtures

D = Decimal


def _e17() -> dict:  # type: ignore[type-arg]
    return load_fixtures()["C"]["E17"]


def _observations(case: dict) -> list[Observation]:  # type: ignore[type-arg]
    return [
        Observation(
            e["security"],
            e["block"],
            {"base": D(e["baseline"]), "cand": D(e["candidate"])},
        )
        for e in case["events"]
    ]


@pytest.mark.parametrize(
    ("block_len", "reps_key", "valid_key", "p_key"),
    [
        (1, "expected_replicates", "expected_B_valid", "expected_p"),
        (2, "expected_replicates_Lb2", "expected_B_valid_Lb2", "expected_p_Lb2"),
    ],
)
def test_E17_bootstrap_reproduces(
    block_len: int, reps_key: str, valid_key: str, p_key: str
) -> None:
    case = _e17()
    reps = bootstrap(
        _observations(case),
        ["base", "cand"],
        case["I"],
        case["J"],
        case["B"],
        case["seed"],
        block_len,
    )
    test = improvement_test(reps, "base", "cand")
    assert len(reps) == len(case[reps_key]) == case["B"]
    for n, (r, exp) in enumerate(zip(reps, case[reps_key], strict=True)):
        assert list(r.w) == exp["w"] and list(r.v) == exp["v"], n
        for key, method in (("base", "base"), ("cand", "cand")):
            want = D(exp[key]) if exp[key] is not None else None
            assert r.means[method] == want, (n, key)
        want_delta = D(exp["delta"]) if exp["delta"] is not None else None
        assert test.deltas[n] == want_delta, n
        assert r.valid is (exp["delta"] is not None)
    assert test.valid == case[valid_key]
    assert test.p == D(case[p_key])
    # 17 §9：B_v < 0.99 B 时方法判为不适用。L_b = 2 时 B_v = 7 < 7.92（回执 E5-01）
    assert test.applicable is (case[valid_key] >= Decimal("0.99") * case["B"])


def test_draw_order_securities_then_blocks() -> None:
    """抽取顺序：先证券、再块；L_b > 1 时起点个数 ceil(J / L_b)，展开后截取前 J 个。"""
    rng, ref = random.Random(7), random.Random(7)
    w, v = draw(rng, 3, 5, 2)
    sec = [ref.randrange(3) for _ in range(3)]
    starts = [ref.randrange(4) for _ in range(3)]  # J − L_b + 1 = 4，ceil(5 / 2) = 3
    blocks = [s + k for s in starts for k in range(2)][:5]
    assert list(w) == [sec.count(i) for i in range(3)]
    assert list(v) == [blocks.count(b) for b in range(5)] and sum(v) == 5


def _rep(base: str | None, cand: str | None) -> Replicate:
    raw = {"base": D(base) if base else None, "cand": D(cand) if cand else None}
    return Replicate((1,), (1,), raw, raw, all(x is not None for x in raw.values()))


def test_bv_threshold_and_p() -> None:
    """B = 100：B_v = 99 恰等于 0.99 B 时适用；98 时方法判为不适用（p 照算照报）。"""
    ok = [_rep("0.5", "0.4")] * 98 + [_rep("0.3", "0.4")] + [_rep(None, "0.4")]
    t = improvement_test(ok, "base", "cand")
    assert t.valid == 99 and t.applicable
    assert t.p == D("0.02")  # (1 + 1) / (99 + 1)
    assert t.lower95 == D("0.1")  # 第 5 百分位：h = 98 × 0.05 = 4.9，排序后第 4、5 个都为 0.1
    bad = [_rep("0.5", "0.4")] * 98 + [_rep(None, "0.4")] * 2
    t = improvement_test(bad, "base", "cand")
    assert t.valid == 98 and not t.applicable
    assert t.p == D("0.010101010101")  # p 照算照报：(1 + 0) / (98 + 1)；方法判为不适用
    none = improvement_test([_rep(None, "0.4")] * 3, "base", "cand")
    assert none.valid == 0 and not none.applicable and none.p is None and none.lower95 is None


def test_holm_standard_step_down() -> None:
    """p = 0.01、0.04、0.03、0.20（m = 4）：0.01 × 4 = 0.04；0.03 × 3 = 0.09；
    0.04 × 2 = 0.08 → 累计最大 0.09；0.20 × 1 = 0.20。"""
    got = holm({"a": D("0.01"), "b": D("0.04"), "c": D("0.03"), "d": D("0.20")})
    assert got == {"a": D("0.04"), "c": D("0.09"), "b": D("0.09"), "d": D("0.2")}
    assert holm({"x": D("0.6"), "y": D("0.7")}) == {"x": D(1), "y": D(1)}  # 上限 1
    assert holm({}) == {}


def test_calendar_blocks_tail_merged() -> None:
    days = tuple(date(2010, 1, 4) + timedelta(days=i) for i in range(130))  # 130 = 63 × 2 + 4
    b = CalendarBlocks(days)
    assert b.count == 2
    assert b.block_of(days[0]) == 0 and b.block_of(days[62]) == 0 and b.block_of(days[63]) == 1
    assert b.block_of(days[129]) == 1  # 末尾 4 日并入前一块
    assert CalendarBlocks(days[:40]).count == 1
    with pytest.raises(InvalidInputError):
        b.block_of(date(2009, 1, 1))


def test_bootstrap_rejects_bad_inputs() -> None:
    obs = [Observation(0, 0, {"m": D("0.1")})]
    with pytest.raises(InvalidInputError):
        bootstrap(obs, ["m"], 1, 2, 10, 1, 3)  # L_b > J
    with pytest.raises(InvalidInputError):
        bootstrap([Observation(1, 0, {"m": D(1)})], ["m"], 1, 1, 10, 1, 1)  # 证券号越界
    with pytest.raises(InvalidInputError):
        bootstrap(obs, ["m"], 1, 1, 0, 1, 1)  # B 须为正
