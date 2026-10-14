import pytest

from stock_risk.domain.numeric import (
    DEFAULT_TOLERANCES,
    InvalidInputError,
    Kind,
    ge,
    gt,
    le,
    lt,
    parse_finite,
    quantize_price,
    recompute_equal,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("96.0000004", 96.000000),
        ("96.0000005", 96.000000),
        ("96.0000015", 96.000002),
        ("96.0000025", 96.000002),
    ],
)
def test_S21g_quantize_half_even(text: str, expected: float) -> None:
    assert quantize_price(text) == expected


def test_quantize_from_text_not_float() -> None:
    # 2.0572899133 → 2.057290（六位，HALF_EVEN）
    assert quantize_price("2.0572899133") == 2.05729
    assert quantize_price(" 100 ") == 100.0


@pytest.mark.parametrize(
    "text", ["nan", "NaN", "inf", "-Infinity", "0", "0.0000004", "-1", "abc", ""]
)
def test_quantize_rejects_non_finite_non_positive_and_garbage(text: str) -> None:
    # "0.0000004" 为正，但量化后为 0，按量化后的值判非法
    with pytest.raises(InvalidInputError):
        quantize_price(text)


def test_parse_finite() -> None:
    assert parse_finite("1086400.0", "volume") == 1086400.0
    assert parse_finite("0", "volume") == 0.0
    with pytest.raises(InvalidInputError):
        parse_finite("inf", "volume")


def test_S21c_start_at_boundary_within_eps_ratio() -> None:
    d_t = (100 - (96 + 1e-10)) / 2  # 2 − 5e-11
    assert ge(d_t, 2, Kind.RATIO)


def test_S21d_no_start_beyond_eps_ratio() -> None:
    d_t = (100 - (96 + 4e-10)) / 2  # 2 − 2e-10
    assert not ge(d_t, 2, Kind.RATIO)


def test_S21e_not_new_low_within_eps_price() -> None:
    assert not lt(80 - 5e-10, 80, Kind.PRICE)


def test_S21f_new_low_beyond_eps_price() -> None:
    assert lt(80 - 2e-9, 80, Kind.PRICE)


def test_comparison_definitions() -> None:
    e = DEFAULT_TOLERANCES.eps_price
    assert gt(80 + 2e-9, 80, Kind.PRICE) and not gt(80 + 0.5 * e, 80, Kind.PRICE)
    assert le(80 + 0.5 * e, 80, Kind.PRICE) and not le(80 + 2e-9, 80, Kind.PRICE)
    assert le(-3.0, -3, Kind.RATIO)  # 13 号 S19c：≤ 含等号


def test_recompute_criterion_examples() -> None:
    # v0.4 第 2.0 节"复算验收判据"的验收例
    assert recompute_equal(0.0, 5e-13, Kind.RATIO)
    assert not recompute_equal(0.0, 2e-12, Kind.RATIO)
    assert recompute_equal(100.0, 100.00000005, Kind.PRICE)
    # 补充的不通过例：允许误差 1e-9 + 1e-9 × 100 ≈ 1.01e-7，差 2e-7 超出
    assert not recompute_equal(100.0, 100.0000002, Kind.PRICE)
