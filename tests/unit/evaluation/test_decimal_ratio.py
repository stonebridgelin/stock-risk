"""D-01、D-02 的精确边界，与引擎 float ε 分开。"""

from decimal import Decimal, localcontext

import pytest

from stock_risk.domain.evaluation.decimal_ratio import decimal_price, drawdown, q12, ratio
from stock_risk.domain.numeric import InvalidInputError


def test_half_even_final_quantization() -> None:
    assert q12(Decimal("0.1999999999995")) == Decimal("0.200000000000")
    assert q12(Decimal("0.2000000000005")) == Decimal("0.200000000000")
    assert q12(Decimal("0.2000000000015")) == Decimal("0.200000000002")
    assert drawdown(Decimal("1600000.000001"), Decimal(2000000)) == Decimal("0.2")
    assert drawdown(Decimal("1600000.000002"), Decimal(2000000)) < Decimal("0.2")


def test_helpers_use_local_context_without_intermediate_rounding() -> None:
    with localcontext() as context:
        context.prec = 3
        context.rounding = "ROUND_DOWN"
        assert ratio(Decimal(1), Decimal(3)) == Decimal("0.333333333333")
        assert drawdown(Decimal(2), Decimal(3)) == Decimal("0.333333333333")
        assert context.prec == 3


def test_price_conversion_checks_original_quantization() -> None:
    """T09-E1R R3：显式检查（不用 assert），python -O 下同样抛错。"""
    assert decimal_price(123.123456) == Decimal("123.123456")
    with pytest.raises(InvalidInputError, match="D-02"):
        decimal_price(1.1234567)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_price_conversion_rejects_invalid_prices(value: float) -> None:
    with pytest.raises(InvalidInputError):
        decimal_price(value)
