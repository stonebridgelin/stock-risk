"""T06b 对照回放的变体全集、套编号、基准与单开关配对（任务卡 T06b"变体"一节）。纯函数。

数值取参考候选（N 20、k₁ 2、k₂ 4、K 3、d 3、u_rel 3），每个变体在 12 种结构组合上运行。
N14 组为 V0 与 V3 参考取值在 N = 14 下的 24 套（网格 1–12）。合计 624 套。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

from stock_risk.domain.lifecycle.params import (
    Candidate,
    LongPath,
    RecScale,
    ReleaseHyst,
    V08Switches,
    candidate_grid,
)

STRUCTURE_N20 = tuple(range(385, 397))  # 参考候选 385–396 的结构
STRUCTURE_N14 = tuple(range(1, 13))  # 网格 1–12：同样 12 种结构，N = 14
V1_GRID = tuple(
    itertools.product((0.08, 0.10), (0.15, 0.20), (LongPath.X1, LongPath.X2), (False, True))
)  # x1 × x2 × long_path × r2_floor
V3_REFERENCE = V08Switches(
    x1=0.08,
    x2=0.15,
    t11_full=True,
    long_path=LongPath.X1,
    r2_floor=False,
    release_hyst=ReleaseHyst.EFFECTIVE,
)


@dataclass(frozen=True, slots=True)
class VariantSet:
    """一套：变体组 × 开关 × 候选（结构组合与 N）。"""

    group: str  # V0 / V1 / V2 / V3 / V4 / N14
    variant: str  # V0 / V1 / V2 / V3 / V4（N14 组内为 V0 或 V3）
    switches: V08Switches
    candidate_id: int  # 网格编号（N20：385–396；N14：1–12）
    candidate: Candidate

    @property
    def n(self) -> int:
        return self.candidate.n

    @property
    def structure_id(self) -> int:
        """对应 N20 参考候选的编号（用于配对）。"""
        return self.candidate_id if self.n == 20 else self.candidate_id + 384

    @property
    def set_id(self) -> str:
        sw = self.switches
        return (
            f"{self.variant}_x{round(sw.x1 * 100)}_x{round(sw.x2 * 100)}_{sw.long_path.value}"
            f"_r2{'on' if sw.r2_floor else 'off'}_N{self.n}_s{self.candidate_id}"
        )

    @property
    def is_baseline(self) -> bool:
        return self.variant == "V0"

    @property
    def baseline_id(self) -> str:
        """同一结构组合、同一 N 的 V0。"""
        sw = V08Switches()
        return f"V0_x0_x0_{sw.long_path.value}_r2off_N{self.n}_s{self.candidate_id}"

    @property
    def is_v3_reference(self) -> bool:
        return self.variant == "V3" and self.switches == V3_REFERENCE


def _v1(x1: float, x2: float, lp: LongPath, r2: bool) -> V08Switches:
    return V08Switches(x1=x1, x2=x2, t11_full=True, long_path=lp, r2_floor=r2)


def variant_sets() -> tuple[VariantSet, ...]:
    grid = dict(candidate_grid())
    out: list[VariantSet] = []

    def add(group: str, variant: str, sw: V08Switches, ids: tuple[int, ...]) -> None:
        out.extend(VariantSet(group, variant, sw, i, grid[i]) for i in ids)

    add("V0", "V0", V08Switches(), STRUCTURE_N20)
    for combo in V1_GRID:
        add("V1", "V1", _v1(*combo), STRUCTURE_N20)
    add("V2", "V2", V08Switches(release_hyst=ReleaseHyst.EFFECTIVE), STRUCTURE_N20)
    for combo in V1_GRID:
        add("V3", "V3", replace(_v1(*combo), release_hyst=ReleaseHyst.EFFECTIVE), STRUCTURE_N20)
    for combo in V1_GRID:
        sw = replace(_v1(*combo), release_hyst=ReleaseHyst.EFFECTIVE, rec_scale=RecScale.PRICE)
        add("V4", "V4", sw, STRUCTURE_N20)
    add("N14", "V0", V08Switches(), STRUCTURE_N14)
    add("N14", "V3", V3_REFERENCE, STRUCTURE_N14)
    ids = [s.set_id for s in out]
    if len(set(ids)) != len(ids):
        raise AssertionError("套编号重复")
    return tuple(out)


def timeline_sets(sets: tuple[VariantSet, ...]) -> tuple[VariantSet, VariantSet]:
    """T06c 两套：V0_s385 与 V3 参考取值的 s385（N = 20）。"""
    v0 = next(s for s in sets if s.is_baseline and s.n == 20 and s.candidate_id == 385)
    v3 = next(s for s in sets if s.is_v3_reference and s.n == 20 and s.candidate_id == 385)
    return v0, v3


@dataclass(frozen=True, slots=True)
class Pair:
    kind: str  # r2_floor / long_path / x1 / x2 / release_hyst / rec_scale / N
    a: str  # 基准一侧的套编号（OFF、X1、8%、15%、OLD、OFF、N20）
    b: str


def pairs(sets: tuple[VariantSet, ...]) -> tuple[Pair, ...]:
    """单开关配对（任务卡 T06b 第 4 项）。"""
    by_key = {(s.variant, s.switches, s.n, s.structure_id): s.set_id for s in sets}
    out: list[Pair] = []

    def pair(kind: str, a: tuple, b: tuple) -> None:
        if a in by_key and b in by_key:
            out.append(Pair(kind, by_key[a], by_key[b]))

    for s in sets:
        if s.group not in ("V1", "V3", "V4") or s.n != 20:
            continue
        sw, key = s.switches, (s.variant, s.n, s.structure_id)
        flips = {
            "r2_floor": (not sw.r2_floor, {"r2_floor": True}),
            "long_path": (sw.long_path is LongPath.X1, {"long_path": LongPath.X2}),
            "x1": (sw.x1 == 0.08, {"x1": 0.10}),
            "x2": (sw.x2 == 0.15, {"x2": 0.20}),
        }
        for kind, (is_a, change) in flips.items():
            if is_a:
                pair(kind, (key[0], sw, *key[1:]), (key[0], replace(sw, **change), *key[1:]))
        if s.variant == "V1":
            pair(
                "release_hyst",
                (s.variant, sw, *key[1:]),
                ("V3", replace(sw, release_hyst=ReleaseHyst.EFFECTIVE), *key[1:]),
            )
        if s.variant == "V3":
            pair(
                "rec_scale",
                (s.variant, sw, *key[1:]),
                ("V4", replace(sw, rec_scale=RecScale.PRICE), *key[1:]),
            )
    for s in sets:
        if s.group == "V0":
            pair(
                "release_hyst",
                ("V0", s.switches, 20, s.structure_id),
                ("V2", V08Switches(release_hyst=ReleaseHyst.EFFECTIVE), 20, s.structure_id),
            )
        if s.group == "N14":
            pair(
                "N",
                (s.variant, s.switches, 20, s.structure_id),
                (s.variant, s.switches, 14, s.structure_id),
            )
    return tuple(out)
