"""17 §3.1 / D-29：波动统计量、第 7 型分位数与波动分组。

- 证券统计量：可评估日 ATR₂₀ / 收盘（各日按 D-01 量化）的中位数；
  偶数个取中间两数的平均，量化到 12 位。
- 分界：第 1/3、2/3 分位（Hyndman–Fan 第 7 型，十进制，量化到 12 位）。
- 归属：≤ b₁ 低组，(b₁, b₂] 中组，> b₂ 高组；验证期新证券不足 63 个可评估日为未归组。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext
from enum import StrEnum

from stock_risk.domain.evaluation.decimal_ratio import EVALUATION_CONTEXT, q12, ratio
from stock_risk.domain.evaluation.series import EvalSeries
from stock_risk.domain.numeric import InvalidInputError

NEW_SECURITY_DAYS = 252  # 17 §3.1：验证期新证券用前 252 个可评估日
NEW_SECURITY_MIN_DAYS = 63  # 17 §3.1：不足 63 个为未归组

with localcontext(EVALUATION_CONTEXT):
    TERTILES = (Decimal(1) / Decimal(3), Decimal(2) / Decimal(3))  # D-29


class VolGroup(StrEnum):
    LOW = "LOW"
    MID = "MID"
    HIGH = "HIGH"
    UNGROUPED = "UNGROUPED"


@dataclass(frozen=True, slots=True)
class Boundaries:
    b1: Decimal
    b2: Decimal

    def __post_init__(self) -> None:
        if self.b1 > self.b2:
            raise InvalidInputError("波动分界须满足 b₁ ≤ b₂")


def median(values: Sequence[Decimal]) -> Decimal:
    """中位数；偶数个取中间两数的平均。结果量化到 12 位（D-29）。"""
    if not values:
        raise InvalidInputError("中位数需要至少一个值")
    ordered = sorted(values)
    n = len(ordered)
    with localcontext(EVALUATION_CONTEXT):
        if n % 2:
            return q12(ordered[n // 2])
        return q12((ordered[n // 2 - 1] + ordered[n // 2]) / Decimal(2))


def quantile_type7(values: Sequence[Decimal], p: Decimal) -> Decimal:
    """Hyndman–Fan 第 7 型：h = (n − 1)p，x_⌊h⌋ + (h − ⌊h⌋)(x_⌊h⌋+1 − x_⌊h⌋)（下标从 0 起）。"""
    if not values:
        raise InvalidInputError("分位数需要至少一个值")
    if not Decimal(0) <= p <= Decimal(1):
        raise InvalidInputError("分位点须在 [0, 1] 内")
    x = sorted(values)
    with localcontext(EVALUATION_CONTEXT):
        h = Decimal(len(x) - 1) * p
        lo = int(h)  # h ≥ 0，向下取整
        if lo + 1 >= len(x):
            return q12(x[-1])
        return q12(x[lo] + (h - Decimal(lo)) * (x[lo + 1] - x[lo]))


def daily_vol_ratios(
    series: EvalSeries, start: date | None = None, end: date | None = None
) -> tuple[Decimal, ...]:
    """[start, end]（含两端；None 为不限）内可评估日的 ATR₂₀ / 收盘，按日期升序。"""
    result: list[Decimal] = []
    for i, day in enumerate(series.calendar.days):
        if not series.mask[i] or (start and day < start) or (end and day > end):
            continue
        atr, close = series.atr20[i], series.closes[i]
        if atr is None or close is None:  # 掩码（≥ 83）保证 ATR 可得
            raise InvalidInputError(f"{day} 在掩码内却没有 ATR20")
        result.append(ratio(atr, close))
    return tuple(result)


def security_stat(ratios: Sequence[Decimal]) -> Decimal | None:
    """开发期证券统计量：全部可评估日比率的中位数；没有可评估日为 None。"""
    return median(ratios) if ratios else None


def new_security_stat(ratios: Sequence[Decimal]) -> Decimal | None:
    """验证期新证券（ratios 按日期升序）：前 252 个；63–251 个用全部；
    不足 63 个为 None（未归组）。"""
    if len(ratios) < NEW_SECURITY_MIN_DAYS:
        return None
    return median(ratios[:NEW_SECURITY_DAYS])


def boundaries(stats: Sequence[Decimal]) -> Boundaries:
    """选参集全部证券统计量的第 1/3 与第 2/3 分位。"""
    return Boundaries(quantile_type7(stats, TERTILES[0]), quantile_type7(stats, TERTILES[1]))


def assign_group(stat: Decimal | None, bounds: Boundaries) -> VolGroup:
    if stat is None:
        return VolGroup.UNGROUPED
    if stat <= bounds.b1:
        return VolGroup.LOW
    return VolGroup.MID if stat <= bounds.b2 else VolGroup.HIGH


__all__ = [
    "Boundaries",
    "TERTILES",
    "VolGroup",
    "assign_group",
    "boundaries",
    "daily_vol_ratios",
    "median",
    "new_security_stat",
    "quantile_type7",
    "security_stat",
]
