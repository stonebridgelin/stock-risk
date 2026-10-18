"""17 §9 / T09 第 5 节 E5：证券 × 日历块双向自助（pigeonhole）、单侧百分位检验、Holm 校正。

严格按 T09 卡 E5 的程序（夹具 E17 依赖它逐位复现），不做预聚合等优化：
1. rng = random.Random(seed)；
2. 每次重复先抽证券 [rng.randrange(I) for _ in range(I)]；再抽块：L_b = 1 时
   [rng.randrange(J) for _ in range(J)]；L_b > 1 时先抽起点
   [rng.randrange(J − L_b + 1) for _ in range(ceil(J / L_b))]，每个起点展开为 L_b 个相邻块，
   按顺序拼接后截取前 J 个；
3. w_i 为证券 i 被抽中的次数，v_b 为块 b 被抽中的次数；格 (i, b) 的权重 w_i × v_b；
4. 事件加权均值 = Σ 权重 × 值 ÷ Σ 权重（十进制）；分母为 0 的重复对该统计量无效；
   基线或候选（参与本次比较的方法）的均值分母为 0，该次重复对这一比较无效
   （2026-10-17 按 T09-E5F 修订：原文"任一方法无效即该次重复无效"）；
5. Δ* = 基线均值 − 候选均值（同一组权重）；p = (1 + #{有效重复中 Δ* ≤ 0}) ÷ (B_v + 1)；
   B_v < 0.99 B 时方法判为不适用；
6. 单侧 95% 下界取 Δ* 的第 5 百分位（第 7 型）；Holm 校正按标准程序；
7. 块划分：按期间的市场交易日历，自期内首个交易日起每 63 个交易日为一块，
   末尾不足 63 日的并入前一块。
   事件按（证券，峰值日所在块）归组；RELEASE、RECOVERY_OBS 按（证券，信号日所在块）。
B、种子、L_b 由调用方传入（登记值），本模块不写默认值。
"""

import math
import random
from bisect import bisect_left
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext

from stock_risk.domain.evaluation.decimal_ratio import EVALUATION_CONTEXT, q12
from stock_risk.domain.evaluation.vol_groups import quantile_type7
from stock_risk.domain.numeric import InvalidInputError

BLOCK_DAYS = 63  # 17 §9：每 63 个交易日为一块
VALID_SHARE = Decimal("0.99")  # 17 §9：B_v < 0.99 B 时方法不适用
LOWER_P = Decimal("0.05")  # 单侧 95% 下界：Δ* 的第 5 百分位


def _positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidInputError(f"{name} 必须为正整数")


# ---- 第 7 步：日历块 ----


@dataclass(frozen=True, slots=True)
class CalendarBlocks:
    """期间的市场交易日历与块划分。块号从 0 起。"""

    days: tuple[date, ...]

    def __post_init__(self) -> None:
        if not self.days or any(a >= b for a, b in zip(self.days, self.days[1:], strict=False)):
            raise InvalidInputError("期间交易日历须非空、严格升序")

    @property
    def count(self) -> int:
        """J：整块数；末尾不足 63 日的并入前一块（全期不足 63 日时为 1 块）。"""
        return max(1, len(self.days) // BLOCK_DAYS)

    def block_of(self, day: date) -> int:
        i = bisect_left(self.days, day)
        if i == len(self.days) or self.days[i] != day:
            raise InvalidInputError(f"{day} 不在期间交易日历内")
        return min(i // BLOCK_DAYS, self.count - 1)


# ---- 第 2、3 步：抽取与权重 ----


def draw(rng: random.Random, n_securities: int, n_blocks: int, block_len: int) -> tuple[
    tuple[int, ...], tuple[int, ...]
]:  # fmt: skip
    """一次重复的 (w, v)。先抽证券，再抽块（顺序影响随机数流，不可调换）。"""
    securities = [rng.randrange(n_securities) for _ in range(n_securities)]
    if block_len == 1:
        blocks = [rng.randrange(n_blocks) for _ in range(n_blocks)]
    else:
        starts = [
            rng.randrange(n_blocks - block_len + 1) for _ in range(math.ceil(n_blocks / block_len))
        ]
        blocks = [s + k for s in starts for k in range(block_len)][:n_blocks]
    sc, bc = Counter(securities), Counter(blocks)
    return (
        tuple(sc[i] for i in range(n_securities)),
        tuple(bc[b] for b in range(n_blocks)),
    )


# ---- 第 4 步：事件加权均值 ----


@dataclass(frozen=True, slots=True)
class Observation:
    """一个评价单位（事件，或 RELEASE、RECOVERY_OBS）：所在格与各方法的值（不纳入时为 None）。"""

    security: int
    block: int
    values: Mapping[str, Decimal | None]


def weighted_mean(
    observations: Sequence[Observation], method: str, w: Sequence[int], v: Sequence[int]
) -> Decimal | None:
    """Σ 权重 × 值 ÷ Σ 权重（十进制，不量化）；分母为 0 时为 None（该重复对此统计量无效）。"""
    num = den = Decimal(0)
    with localcontext(EVALUATION_CONTEXT):
        for o in observations:
            value = o.values.get(method)
            if value is None:
                continue
            weight = Decimal(w[o.security] * v[o.block])
            num += weight * value
            den += weight
        return num / den if den else None


@dataclass(frozen=True, slots=True)
class Replicate:
    w: tuple[int, ...]
    v: tuple[int, ...]
    means: Mapping[str, Decimal | None]  # 量化到 12 位，供报告
    raw_means: Mapping[str, Decimal | None]  # 不量化，供 Δ* 计算
    valid: bool  # 诊断：全部方法的均值都有定义；不得控制任何一对方法的检验（T09-E5F）


def bootstrap(
    observations: Sequence[Observation],
    methods: Sequence[str],
    n_securities: int,
    n_blocks: int,
    replicates: int,
    seed: int,
    block_len: int,
) -> tuple[Replicate, ...]:
    """第 1–4 步。同一次重复中，所有方法（候选、基线、R0）共用同一组权重。"""
    for name, value in (
        ("I", n_securities),
        ("J", n_blocks),
        ("B", replicates),
        ("L_b", block_len),
    ):
        _positive(name, value)
    if block_len > n_blocks:
        raise InvalidInputError("L_b 不得大于块数 J")
    for o in observations:
        if not (0 <= o.security < n_securities and 0 <= o.block < n_blocks):
            raise InvalidInputError(f"观测的格超出范围：{o.security}, {o.block}")
    rng = random.Random(seed)
    out: list[Replicate] = []
    for _ in range(replicates):
        w, v = draw(rng, n_securities, n_blocks, block_len)
        raw = {m: weighted_mean(observations, m, w, v) for m in methods}
        means = {m: (q12(x) if x is not None else None) for m, x in raw.items()}
        out.append(Replicate(w, v, means, raw, all(x is not None for x in raw.values())))
    return tuple(out)


# ---- 第 5、6 步：检验 ----


@dataclass(frozen=True, slots=True)
class ImprovementTest:
    deltas: tuple[Decimal | None, ...]  # 逐次 Δ*（无效重复为 None），量化到 12 位
    valid: int  # B_v
    applicable: bool  # B_v ≥ 0.99 B；不适用时 p 照算、照报，但方法判为不适用（回执 E5-01）
    p: Decimal | None  # 没有有效重复时为空
    lower95: Decimal | None  # Δ* 的第 5 百分位（只作描述）；没有有效重复时为空


def improvement_test(
    replicates: Sequence[Replicate], baseline: str, candidate: str
) -> ImprovementTest:
    """Δ* = 基线均值 − 候选均值；原假设 Δ ≤ 0。

    配对有效重复 = 基线与候选的均值都不为空的重复；B_v、Δ*、p、适用性与下界都只用这同一组重复，
    与其他方法是否有值无关（T09-E5F，FR-01）。Δ* 按重复原顺序输出，配对无效的记为空。"""
    deltas: list[Decimal | None] = []
    for r in replicates:
        b, c = r.raw_means[baseline], r.raw_means[candidate]
        if b is None or c is None:
            deltas.append(None)
            continue
        with localcontext(EVALUATION_CONTEXT):
            deltas.append(q12(b - c))
    valid = [d for d in deltas if d is not None]
    n = len(replicates)
    with localcontext(EVALUATION_CONTEXT):
        applicable = n > 0 and Decimal(len(valid)) >= VALID_SHARE * Decimal(n)
        if not valid:
            return ImprovementTest(tuple(deltas), 0, applicable, None, None)
        hits = sum(1 for d in valid if d <= 0)
        p = q12(Decimal(1 + hits) / Decimal(len(valid) + 1))
    lower = quantile_type7(valid, LOWER_P)
    return ImprovementTest(tuple(deltas), len(valid), applicable, p, lower)


def holm(p_values: Mapping[str, Decimal]) -> dict[str, Decimal]:
    """Holm 校正（逐步向下）：按 p 升序（并列按名称），第 k 个（k 从 1 起）乘以 m − k + 1，
    取累计最大值，上限 1；结果量化到 12 位。"""
    m = len(p_values)
    order = sorted(p_values, key=lambda k: (p_values[k], k))
    out: dict[str, Decimal] = {}
    running = Decimal(0)
    with localcontext(EVALUATION_CONTEXT):
        for k, name in enumerate(order, start=1):
            running = max(running, min(Decimal(1), Decimal(m - k + 1) * p_values[name]))
            out[name] = q12(running)
    return out


__all__ = [
    "BLOCK_DAYS",
    "CalendarBlocks",
    "ImprovementTest",
    "Observation",
    "Replicate",
    "bootstrap",
    "draw",
    "holm",
    "improvement_test",
    "weighted_mean",
]
