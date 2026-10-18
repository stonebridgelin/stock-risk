"""T09-E5F：重采样比较的有效性只取决于参与比较的方法（FR-01）；不完整窗口记未知后的组装链（C1-01）。

期望值来自任务卡（产品经理推导；乙组由 ChatGPT 与产品经理独立核算），
乙组另在测试中按 T09 卡 E5 第 2 步的抽取程序独立复算（不调用待测实现）。
"""

import random
from dataclasses import replace
from decimal import Decimal
from typing import Any

import pytest

from stock_risk.domain.evaluation.aggregate import adverse_rate
from stock_risk.domain.evaluation.release_windows import (
    AdverseThresholds,
    PostWindow,
    post_signal_window,
)
from stock_risk.domain.evaluation.resampling import (
    ImprovementTest,
    Observation,
    bootstrap,
    improvement_test,
)
from tests.unit.evaluation.helpers import load_fixtures, make_series

D = Decimal
BASE, CAND, IRR = "base", "cand", "irr"


def _summary(t: ImprovementTest) -> tuple:  # type: ignore[type-arg]
    return (t.deltas, t.valid, t.p, t.applicable, t.lower95)


def _run(obs, methods, i, j, b, seed, lb):  # type: ignore[no-untyped-def]
    reps = bootstrap(obs, methods, i, j, b, seed, lb)
    return reps, improvement_test(reps, BASE, CAND)


# ---- 第 3 节第 2 项：ChatGPT 的复现例 ----


def test_chatgpt_reproduction_irrelevant_empty_method() -> None:
    seed = 20261017
    two = [Observation(0, 0, {BASE: D("0.5"), CAND: D("0.4")})]
    three = [Observation(0, 0, {BASE: D("0.5"), CAND: D("0.4"), IRR: None})]
    _, t2 = _run(two, [BASE, CAND], 1, 1, 100, seed, 1)
    reps3, t3 = _run(three, [BASE, CAND, IRR], 1, 1, 100, seed, 1)
    assert t2.valid == 100 and t2.applicable
    assert t2.p == D("0.009900990099")  # (1 + 0) / (100 + 1)
    assert t2.deltas == (D("0.100000000000"),) * 100
    assert _summary(t3) == _summary(t2)  # 加入值全为空的无关方法后完全相同
    assert not any(r.valid for r in reps3)  # 诊断字段：全部方法都有值的重复为 0


# ---- 第 3 节第 3 项：不变性（甲：E17；乙：候选部分为空） ----


def _e17_obs(irr: str | None) -> list[Observation]:
    case = load_fixtures()["C"]["E17"]
    out = []
    for n, e in enumerate(case["events"]):
        vals: dict[str, Decimal | None] = {BASE: D(e["baseline"]), CAND: D(e["candidate"])}
        if irr == "valued":
            vals[IRR] = D("0.3")
        elif irr == "empty":
            vals[IRR] = None
        elif irr == "partial":
            vals[IRR] = D("0.3") if n % 2 == 0 else None
        out.append(Observation(e["security"], e["block"], vals))
    return out


@pytest.mark.parametrize("lb", [1, 2])
def test_invariance_group_a_e17(lb: int) -> None:
    case = load_fixtures()["C"]["E17"]
    args = (case["I"], case["J"], case["B"], case["seed"], lb)
    _, ref = _run(_e17_obs(None), [BASE, CAND], *args)
    variants = [
        (_e17_obs("valued"), [BASE, CAND, IRR]),
        (_e17_obs("empty"), [BASE, CAND, IRR]),
        (_e17_obs("partial"), [BASE, CAND, IRR]),
        (_e17_obs(None), [CAND, BASE]),  # 改变传入顺序
        (_e17_obs("empty"), [IRR, CAND, BASE]),
        (_e17_obs("valued"), [BASE, CAND]),  # 删除无关方法（观测中有值，但不参与）
    ]
    for obs, methods in variants:
        _, t = _run(obs, methods, *args)
        assert _summary(t) == _summary(ref), methods
    # E17 原期望不变
    key = "expected_B_valid" if lb == 1 else "expected_B_valid_Lb2"
    assert ref.valid == case[key]


B_OBS = [
    Observation(0, 0, {BASE: D("0.5"), CAND: D("0.4"), IRR: None}),
    Observation(1, 0, {BASE: D("0.5"), CAND: None, IRR: D("0.6")}),
]


def _independent_draws() -> list[list[int]]:
    """按 T09 卡 E5 第 2 步独立复算抽中的证券（I = 2，J = 1，L_b = 1，seed = 7）。"""
    rng = random.Random(7)
    out = []
    for _ in range(100):
        secs = [rng.randrange(2) for _ in range(2)]
        _ = [rng.randrange(1) for _ in range(1)]
        out.append(secs)
    return out


def test_invariance_group_b_candidate_partially_empty() -> None:
    draws = _independent_draws()
    counts = [d.count(1) for d in draws]
    assert (counts.count(0), counts.count(1), counts.count(2)) == (35, 44, 21)  # 卡上的核算
    reps, t = _run(B_OBS, [BASE, CAND, IRR], 2, 1, 100, 7, 1)
    assert t.valid == 79 and sum(r.valid for r in reps) == 44
    expected = tuple(None if c == 2 else D("0.100000000000") for c in counts)
    assert t.deltas == expected  # 两次都抽到证券 1 的 21 次为空，位置逐一核对
    assert t.p == D("0.012500000000")  # 1 / 80
    assert t.lower95 == D("0.100000000000")
    assert not t.applicable  # 79 < 99，p 仍照常报告
    for methods in ([BASE, CAND], [CAND, BASE], [IRR, BASE, CAND], [CAND, IRR, BASE]):
        _, v = _run(B_OBS, methods, 2, 1, 100, 7, 1)
        assert _summary(v) == _summary(t), methods


def test_replicate_valid_is_diagnostic_only() -> None:
    """第 3 节第 4 项：无关方法为空时 Replicate.valid 为假，但不影响比较。"""
    obs = [Observation(0, 0, {BASE: D("0.5"), CAND: D("0.4"), IRR: None})]
    reps, t = _run(obs, [BASE, CAND, IRR], 1, 1, 10, 1, 1)
    assert all(r.valid is False for r in reps)
    assert t.valid == 10 and t.applicable


# ---- 第 2A 节：组装链（EF-05、G-EF-01、RV-02） ----


class Undefined(Exception):
    pass


def assemble(windows: list[Any], *, plus: bool, history: bool = False) -> Decimal | None:
    """卡上第 2A 节的映射规则（写死）。

    history 为真时允许夹具 v1.3 的旧标记"不完整、adverse 为假"。"""
    flags: list[bool] = []
    for w in windows:
        if w.complete:
            if w.adverse is None:
                raise Undefined("完整而不利未知的窗口：本卡不定义")
            flags.append(bool(w.adverse))
            continue
        if w.adverse is False and not history:
            raise AssertionError("修改后输出中出现不完整、adverse 为假的窗口")
        if plus:
            flags.append(True)  # 补足口径：不完整窗口计为不利；主口径不进入分母
    return adverse_rate(flags)


def _b41_windows(sub: str, q_key: str) -> PostWindow:
    case = load_fixtures()["B"]["B41"][sub]
    series = make_series(case)
    p = case["params"]
    th = AdverseThresholds(D(p["y_pct"]), D(p["y_atr"]))
    q = p["Q"] if q_key == "Q" else p["Q_L"]
    return post_signal_window(
        series, series.calendar.days[case["release_day"]], q, D(case["L_ep"]), th
    )


def _rates(q: list[Any], ql: list[Any], history: bool) -> tuple:  # type: ignore[type-arg]
    return (
        assemble(q, plus=False, history=history),
        assemble(ql, plus=False, history=history),
        assemble(q, plus=True, history=history),
        assemble(ql, plus=True, history=history),
    )


EXPECTED_RATES = (D("0"), None, D("0.5"), D("1"))  # p、p_L、p⁺、p_L⁺


def test_assembly_chain_rates_unchanged_by_c1_01() -> None:
    q = [_b41_windows("B41e", "Q"), _b41_windows("B41f", "Q")]
    ql = [_b41_windows("B41e", "Q_L"), _b41_windows("B41f", "Q_L")]
    # 修改后：实际输出中，不完整窗口的 adverse 都不为假，且没有"完整而未知"的窗口
    for w in q + ql:
        assert not (not w.complete and w.adverse is False)
        assert not (w.complete and w.adverse is None)
    assert [w.adverse for w in (q[0], ql[0], ql[1])] == [None, None, None]
    assert _rates(q, ql, history=False) == EXPECTED_RATES
    # 修改前：夹具 v1.3 的旧标记（三处不完整窗口 adverse 为假），只在历史分支允许
    old = lambda w: replace(w, adverse=False) if not w.complete else w  # noqa: E731
    q_old, ql_old = [old(w) for w in q], [old(w) for w in ql]
    assert _rates(q_old, ql_old, history=True) == EXPECTED_RATES
    with pytest.raises(AssertionError):
        assemble(q_old, plus=False)  # 修改后的输出约束不放宽


def test_assembly_incomplete_but_known_adverse() -> None:
    """G-EF-01：不完整但截断前已知不利的窗口，与 B41f 的 Q 窗口一起汇总。"""
    b41f_q = _b41_windows("B41f", "Q")
    known = replace(b41f_q, complete=False, adverse=True, reasons=frozenset({"INCOMPLETE_WINDOW"}))
    assert assemble([known, b41f_q], plus=False) == D("0")  # 只有 B41f 的 Q 完整
    assert assemble([known, b41f_q], plus=True) == D("0.5")


def test_assembly_rejects_complete_unknown() -> None:
    w = replace(_b41_windows("B41f", "Q"), adverse=None)
    with pytest.raises(Undefined):
        assemble([w], plus=False)
