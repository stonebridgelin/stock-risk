"""17 §2.1 / T09 D-01、D-02：独立的十进制计算上下文，不使用引擎 ε。"""

from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)

from stock_risk.domain.numeric import InvalidInputError

# 完整给出全部配置，不继承调用方的当前上下文（T09-E1R R3）
EVALUATION_CONTEXT = Context(
    prec=28,
    rounding=ROUND_HALF_EVEN,
    Emin=-999999,
    Emax=999999,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[InvalidOperation, DivisionByZero, Overflow],
)
RATIO_QUANTUM = Decimal("1e-12")


def q12(value: Decimal) -> Decimal:
    """只量化最终结果；不改变调用方的十进制上下文（D-01）。"""
    with localcontext(EVALUATION_CONTEXT):
        return value.quantize(RATIO_QUANTUM, rounding=ROUND_HALF_EVEN)


def ratio(numerator: Decimal, denominator: Decimal) -> Decimal:
    with localcontext(EVALUATION_CONTEXT):
        return q12(numerator / denominator)


def drawdown(close: Decimal, peak: Decimal) -> Decimal:
    """按原式 1 − C / P 的顺序计算，不能先量化 C / P。"""
    with localcontext(EVALUATION_CONTEXT):
        return q12(Decimal(1) - close / peak)


def decimal_price(value: float) -> Decimal:
    """已入库的六位量化 float → Decimal；禁止静默修正未量化输入。"""
    result = Decimal(f"{value:.6f}")
    if not result.is_finite() or result <= 0:
        raise InvalidInputError("评价价格必须为有限正数")
    if float(result) != value:  # 显式检查，python -O 下同样生效（T09-E1R R3）
        raise InvalidInputError(f"评价价格尚未按六位小数量化（D-02）: {value!r}")
    return result
