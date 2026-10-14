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


@dataclass(frozen=True, slots=True)
class ExplorationAssumptions:
    """探索假设，非发起人批准值（I6 (a)）。"""

    phi: float = 0.5  # 强恢复的收复比例（A1 待批准）
    k_stab: int = 8  # 长期稳定天数（待 SR-M1 登记，须 ≥ 5）
    gap_break: int = 10  # G：历史断点（待 SR-M1 登记，{10, 20} 二选一）


EXPLORATION_ASSUMPTIONS = ExplorationAssumptions()


@dataclass(frozen=True, slots=True)
class EngineParams:
    candidate: Candidate
    assumptions: ExplorationAssumptions
    fixed: FixedValues = FIXED
    tol: Tolerances = field(default=DEFAULT_TOLERANCES)


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
