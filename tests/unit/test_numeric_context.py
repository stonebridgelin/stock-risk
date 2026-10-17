"""T09-E1R R3：价格量化与数值解析使用固定上下文，不受调用方当前十进制上下文影响。"""

from __future__ import annotations

from decimal import ROUND_DOWN, Context, Decimal, Inexact, InvalidOperation, localcontext

import pytest

from stock_risk.domain.evaluation.decimal_ratio import q12
from stock_risk.domain.numeric import InvalidInputError, parse_finite, quantize_price


def test_quantize_ignores_caller_precision() -> None:
    default = quantize_price("123.456789")
    with localcontext(Context(prec=8)):
        assert quantize_price("123.456789") == 123.456789 == default


def test_quantize_ignores_caller_rounding() -> None:
    default = quantize_price("1.0000015")
    with localcontext(Context(prec=8, rounding=ROUND_DOWN)):
        assert quantize_price("1.0000015") == 1.000002 == default  # 按 HALF_EVEN


def test_quantize_out_of_precision_raises() -> None:
    with pytest.raises(InvalidInputError):
        quantize_price("1e30")


def test_parse_finite_rejects_float_overflow() -> None:
    with pytest.raises(InvalidInputError):
        parse_finite("1e10000", "x")
    assert parse_finite("1e300", "x") == 1e300


@pytest.mark.parametrize(
    "ctx",
    [
        Context(Emax=5),
        Context(traps=[Inexact]),  # 只打开 Inexact 的 trap，InvalidOperation 的 trap 关闭
    ],
)
def test_caller_emax_and_traps_do_not_matter(ctx: Context) -> None:
    assert ctx.Emax == 5 or (ctx.traps[Inexact] and not ctx.traps[InvalidOperation])
    default_price = quantize_price("123.456789")
    default_q = q12(Decimal("0.1234567890125"))
    with localcontext(ctx):
        assert quantize_price("123.456789") == default_price
        assert q12(Decimal("0.1234567890125")) == default_q
        with pytest.raises(InvalidInputError):
            quantize_price("1e30")
