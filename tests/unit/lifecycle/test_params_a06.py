"""T08a A-06：Candidate、ExplorationAssumptions 的取值检查；正式组合约束（只供选参使用）。"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from stock_risk.domain.lifecycle.params import (
    AtrBasis,
    Candidate,
    Confirm,
    EngineParams,
    ExplorationAssumptions,
    InvalidParameter,
    LongPath,
    ReleaseHyst,
    ReleasePath,
    V08Switches,
    candidate_grid,
)

BASE = dict(n=20, k1=2, k2=4, k=3, d=3, u_rel=3)
STRUCT = dict(confirm=Confirm.SMA, release_path=ReleasePath.TWO_STAGE, atr_basis=AtrBasis.PEAK)


def cand(**over: object) -> Candidate:
    return Candidate(**{**BASE, **over}, **STRUCT)  # type: ignore[arg-type]


def test_grid_and_defaults_are_valid() -> None:
    assert len(candidate_grid()) == 768
    ExplorationAssumptions()


@pytest.mark.parametrize(
    "over",
    [
        {"n": 0},
        {"n": 20.0},
        {"n": True},
        {"k": -1},
        {"d": 0},
        {"k1": math.nan},
        {"k2": math.inf},
        {"u_rel": math.inf},
        {"u_rel": 0},  # T08d-1 第 4A 节：u_rel > 0
        {"u_rel": -1},
        {"k1": 0},
        {"k1": 4, "k2": 4},
        {"k1": 5, "k2": 4},
    ],
)
def test_candidate_rejects(over: dict) -> None:
    with pytest.raises(InvalidParameter):
        cand(**over)


@pytest.mark.parametrize(
    "over",
    [
        {"phi": 0},
        {"phi": 1.5},
        {"phi": math.nan},
        {"k_stab": 4},
        {"k_stab": 8.0},
        {"gap_break": 0},
        {"gap_break": 2.5},
    ],
)
def test_assumptions_reject(over: dict) -> None:
    with pytest.raises(InvalidParameter):
        ExplorationAssumptions(**over)  # type: ignore[arg-type]


def test_assumptions_boundaries_accepted() -> None:
    ExplorationAssumptions(phi=1, k_stab=5, gap_break=1)


def test_k_stab_not_below_k() -> None:
    a = ExplorationAssumptions(k_stab=5)
    EngineParams(cand(k=5), a)
    with pytest.raises(InvalidParameter, match="K_stab ≥ max"):
        EngineParams(cand(k=6), a)


V3 = V08Switches(
    x1=0.08, x2=0.15, t11_full=True, long_path=LongPath.X1, release_hyst=ReleaseHyst.EFFECTIVE
)


@pytest.mark.parametrize(
    ("switches", "formal"),
    [
        (V08Switches(), True),  # V0
        (V08Switches(release_hyst=ReleaseHyst.EFFECTIVE), True),  # V2
        (V3, True),
        (replace(V3, release_hyst=ReleaseHyst.OLD), False),  # L 配 OLD（V1）
        (replace(V3, long_path=LongPath.OFF, release_hyst=ReleaseHyst.OLD), True),
        (V08Switches(x2=0.15, t11_full=True), False),  # x₂ > 0 而 x₁ = 0
        (V08Switches(x1=0.08, t11_full=True), False),  # t11_full ON 而 x₂ = 0
        (V08Switches(x1=0.08), True),
    ],
)
def test_formal_candidate(switches: V08Switches, formal: bool) -> None:
    p = EngineParams(cand(), ExplorationAssumptions(), switches=switches)
    assert p.is_formal_candidate is formal
    assert (switches.formal_violations(0.5) == ()) is formal


@pytest.mark.parametrize(("h", "formal"), [(0.5, True), (1.0, False)])
def test_formal_candidate_requires_h_below_one(h: float, formal: bool) -> None:
    p = EngineParams(cand(), ExplorationAssumptions(h=h), switches=V3)
    assert p.is_formal_candidate is formal  # h = 1 引擎允许，但不是正式组合
