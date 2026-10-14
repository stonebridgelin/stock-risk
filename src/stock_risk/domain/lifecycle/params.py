"""参数对象（10 号规格第 4 节；03 计划 I6）。

- Candidate：9 个候选维度（第 4.1 节），全集 768 组。
- FixedValues：固定值（第 4.2 节，W、g、SMA 长度）。代码与测试不得再写这些字面值。
- ExplorationAssumptions：探索假设（I6 (a)），**非发起人批准值**：
  φ（A1 待批准）、K_stab、G（待 SR-M1 登记）。
- 数值容差（I6 (b)）复用 domain/numeric.py 的 Tolerances。
参数整体注入引擎，不依赖全局默认值；夹具使用自己的参数（13 号第 1 节，例如 G = 3）。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from enum import StrEnum

from stock_risk.domain.numeric import DEFAULT_TOLERANCES, Tolerances


class Confirm(StrEnum):
    SMA = "SMA"
    BOX = "BOX"
    BOX_ROLL = "BOX_ROLL"


class ReleasePath(StrEnum):
    TWO_STAGE = "TWO_STAGE"
    DIRECT = "DIRECT"


class AtrBasis(StrEnum):
    PEAK = "PEAK"
    MIN = "MIN"


@dataclass(frozen=True, slots=True)
class Candidate:
    """一个候选（第 4.1 节的 9 个维度）。"""

    n: int  # ATR 窗口
    k1: float
    k2: float
    k: int  # 企稳天数 K
    d: int  # 连续天数
    u_rel: float
    confirm: Confirm
    release_path: ReleasePath
    atr_basis: AtrBasis


@dataclass(frozen=True, slots=True)
class FixedValues:
    """第 4.2 节固定值。"""

    w: int = 63  # 回撤窗口
    g: float = 3  # 跳空阈值，只用于启动
    sma_len: int = 10  # 只用于 confirm = SMA


FIXED = FixedValues()


class IllegalCombination(ValueError):
    """v0.8 开关的非法组合（10 号第 1A.1 节）。构造时报错，不自动调整。"""


@dataclass(frozen=True, slots=True)
class ExplorationAssumptions:
    """探索假设，非发起人批准值（I6 (a)）。"""

    phi: float = 0.5  # 强恢复的收复比例（A1 待批准）
    k_stab: int = 8  # 长期稳定天数（待 SR-M1 登记，须 ≥ 5）
    gap_break: int = 10  # G：历史断点（待 SR-M1 登记，{10, 20} 二选一）
    h: float = 0.5  # a′ 回滞比例（v0.8 第 1A.1 节固定探索假设）
    w_long: int = 252  # 长窗口 W_L（v0.8 第 1A.1 节固定探索假设）

    def __post_init__(self) -> None:
        if not (math.isfinite(self.h) and 0 < self.h <= 1):
            raise IllegalCombination(f"违反约束 5：0 < h ≤ 1，实际 h = {self.h}")
        if self.w_long < 1:
            raise IllegalCombination(f"W_L 必须为正整数，实际 {self.w_long}")


EXPLORATION_ASSUMPTIONS = ExplorationAssumptions()


class LongPath(StrEnum):
    OFF = "OFF"
    X1 = "X1"  # L 路径门槛 x_L = x₁
    X2 = "X2"  # x_L = x₂


class ReleaseHyst(StrEnum):
    OLD = "OLD"  # a′ = D_ep < k₁
    EFFECTIVE = "EFFECTIVE"  # a′ = Δ < h × θ（第 1A.5 节）


class RecScale(StrEnum):
    OFF = "OFF"
    PRICE = "PRICE"  # 恢复侧用 ATR_rec = ATR_ep × L_ep / P_ep（第 1A.6 节）


@dataclass(frozen=True, slots=True)
class V08Switches:
    """v0.8 开关（10 号第 1A.1 节）。默认全部关闭（V0），此时与 v0.7.1 逐项相同。

    x1、x2 为比例（0.08 = 8%），0 表示关闭。构造时检查约束 1–4，违反即报错。
    """

    x1: float = 0.0
    x2: float = 0.0
    t11_full: bool = False
    long_path: LongPath = LongPath.OFF
    r2_floor: bool = False
    release_hyst: ReleaseHyst = ReleaseHyst.OLD
    rec_scale: RecScale = RecScale.OFF

    def __post_init__(self) -> None:
        for name in ("x1", "x2"):
            v = getattr(self, name)
            if not (math.isfinite(v) and 0 <= v < 1):
                raise IllegalCombination(f"{name} 必须是 [0, 1) 内的比例，实际 {v}")
        if self.x2 > 0 and not self.t11_full:
            raise IllegalCombination("违反约束 1：x₂ > 0 时 t11_full 必须为 ON")
        if self.long_path is LongPath.X1 and self.x1 == 0:
            raise IllegalCombination("违反约束 2：long_path = X1 时 x₁ 必须 > 0")
        if self.long_path is LongPath.X2 and self.x2 == 0:
            raise IllegalCombination("违反约束 2：long_path = X2 时 x₂ 必须 > 0")
        if self.r2_floor and self.x1 == 0:
            raise IllegalCombination("违反约束 3：r2_floor = ON 时 x₁ 必须 > 0")
        if self.x1 > 0 and self.x2 > 0 and not self.x2 > self.x1:
            raise IllegalCombination("违反约束 4：x₁ > 0 且 x₂ > 0 时必须 x₂ > x₁")

    @property
    def x_long(self) -> float:
        """L 路径门槛 x_L；long_path = OFF 时为 0（不计算长窗口）。"""
        if self.long_path is LongPath.X1:
            return self.x1
        if self.long_path is LongPath.X2:
            return self.x2
        return 0.0

    @property
    def is_v0(self) -> bool:
        return self == V0_SWITCHES


V0_SWITCHES = V08Switches()


@dataclass(frozen=True, slots=True)
class EngineParams:
    candidate: Candidate
    assumptions: ExplorationAssumptions
    fixed: FixedValues = FIXED
    tol: Tolerances = field(default=DEFAULT_TOLERANCES)
    switches: V08Switches = V0_SWITCHES


# 第 4.1 节：维度自上而下、候选值按表中顺序；网格序号按字典序编号 1–768
GRID_DIMENSIONS: tuple[tuple[str, tuple[object, ...]], ...] = (
    ("n", (14, 20)),
    ("k1", (2, 3)),
    ("k2", (4, 5)),
    ("k", (3, 5)),
    ("d", (3, 5)),
    ("u_rel", (3, 5)),
    ("confirm", (Confirm.SMA, Confirm.BOX, Confirm.BOX_ROLL)),
    ("release_path", (ReleasePath.TWO_STAGE, ReleasePath.DIRECT)),
    ("atr_basis", (AtrBasis.PEAK, AtrBasis.MIN)),
)


def candidate_grid() -> tuple[tuple[int, Candidate], ...]:
    """(网格序号, 候选)，序号 1–768。"""
    names = [name for name, _ in GRID_DIMENSIONS]
    combos = itertools.product(*(values for _, values in GRID_DIMENSIONS))
    return tuple(
        (i, Candidate(**dict(zip(names, combo, strict=True))))  # type: ignore[arg-type]
        for i, combo in enumerate(combos, start=1)
    )


# I7：参考组合（数值维度固定）× 全部 12 种结构组合
REFERENCE_NUMERIC = {"n": 20, "k1": 2, "k2": 4, "k": 3, "d": 3, "u_rel": 3}


def reference_candidates() -> tuple[tuple[int, Candidate], ...]:
    """12 组参考候选，用网格编号标识（I7）。"""
    return tuple(
        (i, c)
        for i, c in candidate_grid()
        if all(getattr(c, k) == v for k, v in REFERENCE_NUMERIC.items())
    )
