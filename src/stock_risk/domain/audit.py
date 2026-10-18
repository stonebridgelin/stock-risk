"""原始数据审计规则（T00 第 4 节）。纯函数，只报告，不修改数据。

- 明确失败（v0.4 §2.0）：非有限值；高 < 低；收盘不在 [低, 高]；价格 ≤ 0。原始价与复权价分别检查。
- 复权系数跳变：adj_close / close 相对前一日变化超过阈值，且当天没有公司行动记录。
- 交易日缺口：日历上、检查区间内没有日线的交易日。
- 零成交量、封存越界行数。
- MANIFEST 校验：按清单逐个比较日线文件的 SHA-256（T00a）。
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext
from enum import StrEnum

from stock_risk.domain.adjustment.model import GroupRecord, GroupStatus
from stock_risk.domain.market_data import DailyBar
from stock_risk.domain.numeric import PARSE_CONTEXT
from stock_risk.domain.seal import EXPLORATION_END

# 复权系数跳变阈值：任务卡 T00 第 4 节给定（审计报告阈值，不是信号参数）
ADJ_FACTOR_JUMP_THRESHOLD = 0.02


class PriceBasis(StrEnum):
    RAW = "raw"
    ADJ = "adj"


class FailureKind(StrEnum):
    NON_FINITE = "non_finite"
    HIGH_BELOW_LOW = "high_below_low"
    CLOSE_OUTSIDE_RANGE = "close_outside_range"
    NON_POSITIVE_PRICE = "non_positive_price"


@dataclass(frozen=True, slots=True)
class ExplicitFailure:
    day: date
    basis: PriceBasis | None  # 成交量非有限时为 None
    kind: FailureKind
    detail: str


def _prices(b: DailyBar, basis: PriceBasis) -> tuple[float, float, float, float]:
    if basis is PriceBasis.RAW:
        return b.open, b.high, b.low, b.close
    return b.adj_open, b.adj_high, b.adj_low, b.adj_close


def explicit_failures(bars: Iterable[DailyBar]) -> list[ExplicitFailure]:
    """逐日检查 v0.4 §2.0 的“明确失败”。一根日线可能同时命中多条。"""
    out: list[ExplicitFailure] = []
    for b in bars:
        if not math.isfinite(b.volume):
            out.append(ExplicitFailure(b.day, None, FailureKind.NON_FINITE, f"volume={b.volume}"))
        for basis in PriceBasis:
            o, h, lo, c = _prices(b, basis)
            if not all(math.isfinite(v) for v in (o, h, lo, c)):
                out.append(
                    ExplicitFailure(
                        b.day, basis, FailureKind.NON_FINITE, f"O/H/L/C={o}/{h}/{lo}/{c}"
                    )
                )
                continue
            if min(o, h, lo, c) <= 0:
                out.append(
                    ExplicitFailure(
                        b.day, basis, FailureKind.NON_POSITIVE_PRICE, f"O/H/L/C={o}/{h}/{lo}/{c}"
                    )
                )
            if h < lo:
                out.append(
                    ExplicitFailure(b.day, basis, FailureKind.HIGH_BELOW_LOW, f"H={h} L={lo}")
                )
            if not lo <= c <= h:
                out.append(
                    ExplicitFailure(
                        b.day, basis, FailureKind.CLOSE_OUTSIDE_RANGE, f"C={c} L={lo} H={h}"
                    )
                )
    return out


@dataclass(frozen=True, slots=True)
class AdjFactorJump:
    day: date
    prev_day: date
    prev_factor: float
    factor: float
    change: float


def adj_factor_jumps(
    bars: Sequence[DailyBar],
    action_days: Collection[date],
    threshold: float = ADJ_FACTOR_JUMP_THRESHOLD,
) -> list[AdjFactorJump]:
    """复权系数 adj_close / close 相对前一根日线变化超过 threshold，且当天没有公司行动。

    收盘非正或非有限的日线无法计算系数，跳过（已由明确失败报告）。
    """
    out: list[AdjFactorJump] = []
    prev: tuple[date, float] | None = None
    for b in bars:
        if not (math.isfinite(b.close) and math.isfinite(b.adj_close)) or b.close <= 0:
            prev = None
            continue
        f = b.adj_close / b.close
        if prev is not None and prev[1] > 0:
            change = f / prev[1] - 1
            if abs(change) > threshold and b.day not in action_days:
                out.append(AdjFactorJump(b.day, prev[0], prev[1], f, change))
        prev = (b.day, f)
    return out


# ---- 规范数据的区间审计（T11-D0a 第 8 节；规格第 9.2 节） ----

CANONICAL_EPSILON = Decimal("0.0000005")  # F 量化到 6 位的误差界 ε = 5 × 10⁻⁷

_Ratio = tuple[int, int]  # 精确有理数 (分子, 分母)，分母为正


def _ratio(x: Decimal) -> _Ratio:
    """十进制 → 精确有理数（Decimal.as_integer_ratio 是精确的，不经过 float）。"""
    return x.as_integer_ratio()


def _mul(*xs: _Ratio) -> _Ratio:
    n, d = 1, 1
    for a, b in xs:
        n, d = n * a, d * b
    return n, d


def _add(x: _Ratio, y: _Ratio) -> _Ratio:
    return x[0] * y[1] + y[0] * x[1], x[1] * y[1]


def _neg(x: _Ratio) -> _Ratio:
    return -x[0], x[1]


def _le(x: _Ratio, y: _Ratio) -> bool:
    """x ≤ y（分母为正，交叉相乘）。"""
    return x[0] * y[1] <= y[0] * x[1]


def factor_change_within(
    r_prev: Decimal,
    f_prev: Decimal,
    r_day: Decimal,
    f_day: Decimal,
    delta_b: _Ratio,
    epsilon: Decimal = CANONICAL_EPSILON,
) -> bool:
    """ΔB 是否落在允许区间：
    [(F_prev − ε)·R_day ÷ ((F_day + ε)·R_prev)，(F_prev + ε)·R_day ÷ ((F_day − ε)·R_prev)]。

    方向：F = R ÷ B，所以 (R_day/F_day) ÷ (R_prev/F_prev) = ΔB（与 adj_factor_jumps 相反）。
    全部转成精确有理数后交叉相乘比较（T11R-01），不在任何有限精度上下文中做除法。"""
    rp, fp, rd, fd, e = (_ratio(x) for x in (r_prev, f_prev, r_day, f_day, epsilon))
    # 下界：(F_prev − ε)·R_day ≤ ΔB·(F_day + ε)·R_prev
    lower_ok = _le(_mul(_add(fp, _neg(e)), rd), _mul(delta_b, _add(fd, e), rp))
    # 上界：ΔB·(F_day − ε)·R_prev ≤ (F_prev + ε)·R_day（F_day − ε > 0）
    fd_minus = _add(fd, _neg(e))
    if fd_minus[0] <= 0:
        return lower_ok
    upper_ok = _le(_mul(delta_b, fd_minus, rp), _mul(_add(fp, e), rd))
    return lower_ok and upper_ok


@dataclass(frozen=True, slots=True)
class CanonicalBarPrices:
    """区间审计所需的一根日线：原始收盘 R 与 F 收盘（十进制，来自规范数据的文本）。"""

    day: date
    raw_close: Decimal
    f_close: Decimal


@dataclass(frozen=True, slots=True)
class CanonicalFactorMismatch:
    prev_day: date
    day: date
    expected_factor: Decimal  # 区间内 APPLIED 组的因子之积 ΔB
    groups: tuple[str, ...]


def canonical_factor_checks(
    bars: Sequence[CanonicalBarPrices], groups: Sequence[GroupRecord]
) -> list[CanonicalFactorMismatch]:
    """对前后两根有效日线 (prev_day, day]，取计算组表中 e′ 落在区间内的 APPLIED 组各乘一次，
    得到预期累计因子 ΔB（区间内没有组时为 1），核对 R/F 的比值变化；不按原始行动相乘（T11-03）。"""
    applied = sorted(
        (g for g in groups if g.status is GroupStatus.APPLIED and g.effective_day is not None),
        key=lambda g: (g.effective_day, g.original_ex_date, g.group_id),
    )
    out: list[CanonicalFactorMismatch] = []
    for prev, cur in zip(bars, bars[1:], strict=False):
        inside = [g for g in applied if prev.day < g.effective_day <= cur.day]  # type: ignore[operator]
        ratio: _Ratio = (1, 1)
        expected = Decimal(1)
        for g in inside:
            assert g.b is not None
            ratio = _mul(ratio, _ratio(g.b))
            with localcontext(PARSE_CONTEXT):  # 只用于报告；判断用上面的精确有理数
                expected = expected * g.b
        if not factor_change_within(
            prev.raw_close, prev.f_close, cur.raw_close, cur.f_close, ratio
        ):
            out.append(
                CanonicalFactorMismatch(
                    prev.day, cur.day, expected, tuple(g.group_id for g in inside)
                )
            )
    return out


def calendar_gaps(
    bars: Sequence[DailyBar],
    calendar: Sequence[date],
    start: date | None = None,
    end: date | None = None,
) -> list[date]:
    """检查区间内日历上没有日线的交易日。

    区间默认取首条至末条日线；给定 start / end（上市期间）时以其为界。
    区间同时受日历覆盖范围限制。
    """
    if not bars and (start is None or end is None):
        return []
    lo = start if start is not None else bars[0].day
    hi = end if end is not None else bars[-1].day
    have = {b.day for b in bars}
    return [d for d in calendar if lo <= d <= hi and d not in have]


def zero_volume_days(bars: Iterable[DailyBar]) -> list[date]:
    return [b.day for b in bars if b.volume == 0]


class ManifestStatus(StrEnum):
    HASH_MISMATCH = "hash_mismatch"  # 文件哈希与清单不一致
    FILE_MISSING = "file_missing"  # 清单中有、目录中没有
    NOT_IN_MANIFEST = "not_in_manifest"  # 目录中有、清单中没有


@dataclass(frozen=True, slots=True)
class ManifestMismatch:
    file: str
    status: ManifestStatus
    expected: str | None
    actual: str | None


def manifest_mismatches(
    expected: Mapping[str, str], actual: Mapping[str, str]
) -> list[ManifestMismatch]:
    """按 MANIFEST 逐个比较日线文件的 SHA-256。expected / actual：文件名 → 哈希。"""
    out: list[ManifestMismatch] = []
    for name in sorted(set(expected) | set(actual)):
        e, a = expected.get(name), actual.get(name)
        if e is None:
            out.append(ManifestMismatch(name, ManifestStatus.NOT_IN_MANIFEST, None, a))
        elif a is None:
            out.append(ManifestMismatch(name, ManifestStatus.FILE_MISSING, e, None))
        elif e.lower() != a.lower():
            out.append(ManifestMismatch(name, ManifestStatus.HASH_MISMATCH, e, a))
    return out


def count_sealed_rows(days: Iterable[date]) -> int:
    """晚于探索期截止日的行数（应为 0）。只计数，不读取其数值。"""
    return sum(1 for d in days if d > EXPLORATION_END)
