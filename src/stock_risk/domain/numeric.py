"""数值规范（I9；v0.4 第 2.0 节）：价格量化、ε 比较、复算判据。

- 价格量化：输入为文本，按 Decimal 解析，ROUND_HALF_EVEN 量化到 6 位小数，再转为 float。
- 规则比较：x ≥ k 判为 x ≥ k − ε；x > k 判为 x > k + ε；x ≤ k 判为 x ≤ k + ε；x < k 判为 x < k − ε。
  比率类 ε_r、价格类 ε_p 两套。
- 复算判据：|a − b| ≤ abs_tol + rel_tol × max(|a|, |b|)，与规则比较容差分开。
- 求和统一用 math.fsum，输入按交易日升序传入（由调用方保证）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from enum import StrEnum


class InvalidInputError(ValueError):
    """输入违反 v0.4 第 2.0 节的"明确失败"规则（非有限值、非正价格、高 < 低等）。整次运行失败。"""


PRICE_QUANTUM = Decimal("0.000001")  # 六位小数（v0.4 第 2.0 节"入库量化"）


@dataclass(frozen=True, slots=True)
class Tolerances:
    """数值容差，待登记（I6 (b)）。值取自 v0.4 第 2.0 节。"""

    eps_ratio: float = 1e-10  # ε_r：比率类规则比较
    eps_price: float = 1e-9  # ε_p：价格类规则比较（美元）
    recompute_abs_ratio: float = 1e-12  # 复算判据：比率类 abs_tol
    recompute_abs_price: float = 1e-9  # 复算判据：价格类 abs_tol
    recompute_rel: float = 1e-9  # 复算判据：相对项系数


DEFAULT_TOLERANCES = Tolerances()


class Kind(StrEnum):
    RATIO = "ratio"
    PRICE = "price"


def _decimal(text: str, field: str) -> Decimal:
    try:
        value = Decimal(text.strip())
    except (InvalidOperation, AttributeError) as e:
        raise InvalidInputError(f"{field} 不是数值: {text!r}") from e
    if not value.is_finite():
        raise InvalidInputError(f"{field} 为非有限值: {text!r}")
    return value


def quantize_price(text: str, field: str = "price") -> float:
    """价格文本 → 六位小数（ROUND_HALF_EVEN）→ float。

    非有限值 → InvalidInputError；量化后 ≤ 0 → InvalidInputError（v0.4 第 2.0 节按量化后的值判定，
    例如 "0.0000004" 量化为 0，属于非法值）。
    """
    value = _decimal(text, field).quantize(PRICE_QUANTUM, rounding=ROUND_HALF_EVEN)
    if value <= 0:
        raise InvalidInputError(f"{field} 量化后不是正数: {text!r}")
    return float(value)


def parse_finite(text: str, field: str) -> float:
    """非价格数值（成交量、公司行动数值）：只要求有限，不量化。"""
    return float(_decimal(text, field))


def _eps(kind: Kind, tol: Tolerances) -> float:
    return tol.eps_ratio if kind is Kind.RATIO else tol.eps_price


def ge(x: float, k: float, kind: Kind, tol: Tolerances = DEFAULT_TOLERANCES) -> bool:
    """x ≥ k，判为 x ≥ k − ε。"""
    return x >= k - _eps(kind, tol)


def gt(x: float, k: float, kind: Kind, tol: Tolerances = DEFAULT_TOLERANCES) -> bool:
    """x > k，判为 x > k + ε。"""
    return x > k + _eps(kind, tol)


def le(x: float, k: float, kind: Kind, tol: Tolerances = DEFAULT_TOLERANCES) -> bool:
    """x ≤ k，判为 x ≤ k + ε。"""
    return x <= k + _eps(kind, tol)


def lt(x: float, k: float, kind: Kind, tol: Tolerances = DEFAULT_TOLERANCES) -> bool:
    """x < k，判为 x < k − ε。"""
    return x < k - _eps(kind, tol)


def recompute_equal(a: float, b: float, kind: Kind, tol: Tolerances = DEFAULT_TOLERANCES) -> bool:
    """复算验收判据：|a − b| ≤ abs_tol + rel_tol × max(|a|, |b|)。"""
    abs_tol = tol.recompute_abs_ratio if kind is Kind.RATIO else tol.recompute_abs_price
    return abs(a - b) <= abs_tol + tol.recompute_rel * max(abs(a), abs(b))


def require_finite(values: tuple[float, ...] | list[float], what: str) -> None:
    if not all(math.isfinite(v) for v in values):
        raise InvalidInputError(f"{what} 含非有限值")
