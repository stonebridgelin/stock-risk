"""17 §5.2：资格、稳健性（相邻候选）与逐层排序。纯函数。

- 组合约束以 T08a 的实现为准：V08Switches 构造时的合法组合检查（违反即报错），
  加上 EngineParams.is_formal_candidate（formal_violations：t11_full ⇔ x₂ > 0、x₂ > 0 ⇒ x₁ > 0、
  long_path ≠ OFF ⇒ release_hyst = EFFECTIVE、0 < h < 1）；
- 资格 = 组合约束 ∧ H1 违反 0 次 ∧ 门槛全部通过 ∧ 证据充分 ∧ 稳健；
- 稳健性：每个满足组合约束的相邻候选须门槛全部通过、证据充分，且 S1、S2 不劣于本候选超过 Δ₁、Δ₂
  （恰等于 Δ 时通过）；不要求相邻候选满足 H1；违反组合约束的相邻候选跳过；
- 排序：A₁ = {S1 ≤ S1* + ε₁}，A₂ = A₁ 内 {S2 ≤ S2* + ε₂}，再按 F 升序、网格序号升序取第一个。
  S1、S2 均值为 E_pct 与 E_atr 的平均（主口径），量化到 12 位。
ε₁、ε₂、Δ₁、Δ₂ 与网格由调用方传入，本模块不写默认值。
"""

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext

from stock_risk.domain.evaluation.decimal_ratio import EVALUATION_CONTEXT, q12
from stock_risk.domain.lifecycle.params import EngineParams
from stock_risk.domain.numeric import InvalidInputError

# 17 §5.2 第 5 项：相邻候选改变的数值维度与阈值维度（结构维度与规则版本不变）
NEIGHBOR_DIMENSIONS = ("n", "k1", "k2", "k", "d", "u_rel", "x1", "x2", "h")


def formal_violations(params: EngineParams) -> tuple[str, ...]:
    """正式组合约束的违反项（空 = 满足）。合法组合约束在 V08Switches 构造时已检查。"""
    return params.switches.formal_violations(params.assumptions.h)


@dataclass(frozen=True, slots=True)
class EligibilityInput:
    formal: bool  # 组合约束（formal_violations 为空）
    h1_violations: int
    gates_pass: bool  # GateSummary.all_pass
    evidence_ok: bool  # GateSummary.evidence_ok
    robust: bool


@dataclass(frozen=True, slots=True)
class Eligibility:
    eligible: bool
    reasons: tuple[str, ...]  # 不合格的原因（空 = 合格）


def eligibility(x: EligibilityInput) -> Eligibility:
    reasons = [
        name
        for name, ok in (
            ("FORMAL", x.formal),
            ("H1", x.h1_violations == 0),
            ("GATES", x.gates_pass),
            ("EVIDENCE", x.evidence_ok),
            ("ROBUSTNESS", x.robust),
        )
        if not ok
    ]
    return Eligibility(not reasons, tuple(reasons))


# ---- 相邻候选 ----


def neighbor_points(
    point: Mapping[str, Hashable],
    grid: Mapping[str, Sequence[Hashable]],
    dimensions: Sequence[str] = NEIGHBOR_DIMENSIONS,
) -> list[dict[str, Hashable]]:
    """每个维度各自改取网格中的相邻值（边界只有一个相邻值），其余维度不变。
    grid 为"维度 → 有序取值"；point 中不在 grid 的维度（结构维度等）原样保留。
    只在 grid 中取值唯一的维度没有相邻值。"""
    out: list[dict[str, Hashable]] = []
    for dim in dimensions:
        if dim not in grid:
            continue
        values = list(grid[dim])
        if point[dim] not in values:
            raise InvalidInputError(f"{dim} = {point[dim]!r} 不在网格中")
        i = values.index(point[dim])
        for j in (i - 1, i + 1):
            if 0 <= j < len(values):
                out.append({**point, dim: values[j]})
    return out


@dataclass(frozen=True, slots=True)
class NeighborResult:
    neighbor_id: str
    formal: bool
    gates_pass: bool
    evidence_ok: bool
    s1: Decimal
    s2: Decimal


@dataclass(frozen=True, slots=True)
class Robustness:
    robust: bool
    skipped: tuple[str, ...]  # 违反组合约束而跳过的相邻候选
    failures: tuple[tuple[str, str], ...]  # (相邻候选, 原因)


def robustness(
    s1: Decimal,
    s2: Decimal,
    neighbors: Sequence[NeighborResult],
    delta1: Decimal,
    delta2: Decimal,
) -> Robustness:
    """17 §5.2 第 5 项。S1、S2 越小越好；退化 = 相邻候选 − 本候选，恰等于 Δ 时通过。"""
    skipped: list[str] = []
    failures: list[tuple[str, str]] = []
    for nb in neighbors:
        if not nb.formal:
            skipped.append(nb.neighbor_id)
            continue
        with localcontext(EVALUATION_CONTEXT):
            checks = (
                ("GATES", nb.gates_pass),
                ("EVIDENCE", nb.evidence_ok),
                ("S1", nb.s1 - s1 <= delta1),
                ("S2", nb.s2 - s2 <= delta2),
            )
        failures.extend((nb.neighbor_id, name) for name, ok in checks if not ok)
    return Robustness(not failures, tuple(skipped), tuple(failures))


# ---- 逐层排序 ----


@dataclass(frozen=True, slots=True)
class RankInput:
    grid: int  # 网格序号
    s1_pct: Decimal
    s1_atr: Decimal
    s2_pct: Decimal
    s2_atr: Decimal
    f: Decimal  # 向上事件频次（START + ESCALATE，每证券年，跨证券中位数）


@dataclass(frozen=True, slots=True)
class Ranking:
    s1_mean: Mapping[int, Decimal]
    s2_mean: Mapping[int, Decimal]
    s1_star: Decimal | None
    a1: tuple[int, ...]
    s2_star: Decimal | None
    a2: tuple[int, ...]
    order: tuple[int, ...]  # A₂ 内按 F、网格序号的顺序
    selected: int | None  # 合格集合为空时为空：停止，请发起人裁决


def _mean(a: Decimal, b: Decimal) -> Decimal:
    with localcontext(EVALUATION_CONTEXT):
        return q12((a + b) / Decimal(2))


def rank(eligible: Sequence[RankInput], eps1: Decimal, eps2: Decimal) -> Ranking:
    """只对合格集合排序（调用方先按 eligibility 筛选）。"""
    if len({c.grid for c in eligible}) != len(eligible):
        raise InvalidInputError("网格序号重复")
    s1 = {c.grid: _mean(c.s1_pct, c.s1_atr) for c in eligible}
    s2 = {c.grid: _mean(c.s2_pct, c.s2_atr) for c in eligible}
    if not eligible:
        return Ranking(s1, s2, None, (), None, (), (), None)
    with localcontext(EVALUATION_CONTEXT):
        s1_star = min(s1.values())
        a1 = tuple(sorted(g for g, v in s1.items() if v <= s1_star + eps1))
        s2_star = min(s2[g] for g in a1)
        a2 = tuple(g for g in a1 if s2[g] <= s2_star + eps2)
    f = {c.grid: c.f for c in eligible}
    order = tuple(sorted(a2, key=lambda g: (f[g], g)))
    return Ranking(s1, s2, s1_star, a1, s2_star, a2, order, order[0])


__all__ = [
    "NEIGHBOR_DIMENSIONS",
    "Eligibility",
    "EligibilityInput",
    "NeighborResult",
    "RankInput",
    "Ranking",
    "Robustness",
    "eligibility",
    "formal_violations",
    "neighbor_points",
    "rank",
    "robustness",
]
