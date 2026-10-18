"""T11-D0a 第 9.2 节的领域边界测试（B1–B7、B9–B12、B17–B20、B23、B25、B27、B28）。

期望值照抄任务卡第 4.5、9.2 节；B8 见 test_adj_volume.py，B13–B16、B21、B22、B24、B26 见
test_canonical_build.py。
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, localcontext

import pytest

from stock_risk.domain.adjustment import view_as_of
from stock_risk.domain.adjustment.inputs import AdjustmentInputError, dividend_from_text
from stock_risk.domain.adjustment.model import Eligibility, GroupStatus
from stock_risk.domain.audit import factor_change_within
from stock_risk.domain.numeric import PARSE_CONTEXT
from tests.unit.adjustment.helpers import (
    DIV_SHA,
    DIV_SOURCE,
    assumption_codes,
    bar,
    day,
    div,
    f_close,
    group,
    group_id_div,
    group_id_split,
    issue_codes,
    run,
    split,
    verification,
    view_bytes,
)
from tests.unit.adjustment.prefix import check_prefix

USABLE, NOT_USABLE = Eligibility.USABLE, Eligibility.NOT_USABLE
D = date


# ---- B1–B3、B27：缺价替代的计数基准（第 4.5 节） ----


def test_b1_fallback_exactly_five_days() -> None:
    bars = [bar(D(2015, 3, 2), 50), bar(D(2015, 3, 10), 49)]  # 3、4、5、6、9 日缺价
    dv = [div(D(2015, 3, 10), "1")]
    r = run(bars, (), dv)
    g = group(r, group_id_div(D(2015, 3, 10)))
    assert g.k == 5 and g.b == Decimal("0.98") and g.status is GroupStatus.APPLIED
    assert f_close(r)[-1] == "50.000000"
    ((a,),) = [[a for a in r.assumptions if a.code == "DIV_PREV_FALLBACK"]]
    assert a.k == 5
    assert r.eligibility is USABLE
    check_prefix(bars, (), dv)


def test_b2_fallback_six_days_too_far() -> None:
    bars = [bar(D(2015, 3, 2), 50), bar(D(2015, 3, 11), 49)]  # 3 月 10 日也缺价
    dv = [div(D(2015, 3, 10), "1")]
    r = run(bars, (), dv)
    g = group(r, group_id_div(D(2015, 3, 10)))
    assert g.k == 6 and g.status is GroupStatus.REJECTED
    assert issue_codes(r) == ["FALLBACK_TOO_FAR"]
    assert r.eligibility is NOT_USABLE and r.reasons == ("FALLBACK_TOO_FAR",)
    check_prefix(bars, (), dv)


def test_b3_weekend_ex_date_k_zero() -> None:
    bars = [bar(D(2015, 3, 6), 50), bar(D(2015, 3, 9), 49)]
    dv = [div(D(2015, 3, 7), "1.00")]  # 周六
    r = run(bars, (), dv)
    g = group(r, group_id_div(D(2015, 3, 7)))
    assert g.k == 0 and g.b == Decimal("0.98") and g.effective_day == D(2015, 3, 9)
    assert assumption_codes(r) == ["ACTION_DATE_SHIFTED"]  # 只记平移
    assert f_close(r)[-1] == "50.000000"
    assert r.eligibility is USABLE
    check_prefix(bars, (), dv)


def test_b27_sequential_theoretical_reference_k_zero() -> None:
    bars = [bar(D(2015, 3, 6), 100), bar(D(2015, 3, 9), 97)]
    dv = [div(D(2015, 3, 7), "1.00", 1), div(D(2015, 3, 9), "2.00", 2)]
    r = run(bars, (), dv)
    g1, g2 = group(r, group_id_div(D(2015, 3, 7))), group(r, group_id_div(D(2015, 3, 9)))
    assert g1.effective_day == g2.effective_day == D(2015, 3, 9)
    assert g1.b == Decimal("0.99")
    assert g2.c_ref == Decimal(99) and g2.reference_is_theoretical is True and g2.k == 0
    with localcontext(PARSE_CONTEXT):
        assert g2.b == Decimal(97) / Decimal(99)
        assert r.bars[1].b_cum.quantize(Decimal("1e-20")) == Decimal("0.97")
    assert f_close(r)[-1] == "100.000000"
    assert g1.codes == ("ACTION_DATE_SHIFTED",)
    assert g2.codes == ("DIV_PREV_FALLBACK",)
    assert r.eligibility is USABLE
    check_prefix(bars, (), dv)


# ---- B4–B6：异常与非法因子（裁决 A-4） ----


def test_b4_factor_exactly_half_is_not_abnormal() -> None:
    bars = [bar(day(1), 10), bar(day(2), 5)]
    r = run(bars, (), [div(day(2), "5")])
    assert group(r, group_id_div(day(2))).b == Decimal("0.5")
    assert f_close(r)[-1] == "10.000000"  # F = R ÷ 0.5
    assert issue_codes(r) == [] and r.eligibility is USABLE
    check_prefix(bars, (), [div(day(2), "5")])


def test_b5_factor_049_abnormal_then_confirmed() -> None:
    bars = [bar(day(1), 100), bar(day(2), 49)]
    dv = [div(day(2), "51")]
    r = run(bars, (), dv)
    assert group(r, group_id_div(day(2))).b == Decimal("0.49")
    assert issue_codes(r) == ["ABNORMAL_FACTOR"] and r.eligibility is NOT_USABLE
    v = [verification("V9", "group", group_id_div(day(2)), "ABNORMAL_CONFIRMED", "51")]
    ok = run(bars, (), dv, v)
    g = group(ok, group_id_div(day(2)))
    assert g.status is GroupStatus.APPLIED and g.verification_ids == ("V9",)
    assert issue_codes(ok) == [] and ok.eligibility is USABLE
    assert f_close(ok)[-1] == "100.000000"
    check_prefix(bars, (), dv, v)
    with pytest.raises(AdjustmentInputError, match="不一致"):  # 登记金额与输入不一致
        run(
            bars,
            (),
            dv,
            [verification("V9", "group", group_id_div(day(2)), "ABNORMAL_CONFIRMED", "50")],
        )


@pytest.mark.parametrize("amount", ["10", "12"])
def test_b6_illegal_factor_never_released(amount: str) -> None:
    bars = [bar(day(1), 10), bar(day(2), 1)]
    dv = [div(day(2), amount)]
    r = run(bars, (), dv)
    assert group(r, group_id_div(day(2))).b == Decimal(1) - Decimal(amount) / Decimal(10)
    assert issue_codes(r) == ["ILLEGAL_FACTOR"] and r.eligibility is NOT_USABLE
    v = [verification("V1", "group", group_id_div(day(2)), "ABNORMAL_CONFIRMED", amount)]
    r2 = run(bars, (), dv, v)
    assert issue_codes(r2) == ["ILLEGAL_FACTOR"] and r2.eligibility is NOT_USABLE
    check_prefix(bars, (), dv, v)


# ---- B7、B25：可表示性（裁决 A-6；T11-07） ----


@pytest.mark.parametrize(
    ("price", "ok"),
    [
        ("0.010000", True),
        ("1000000000.000000", True),
        ("0.009999", False),
        ("1000000000.000001", False),
    ],
)
def test_b7_price_bounds_inclusive(price: str, ok: bool) -> None:
    r = run([bar(day(1), price)])
    assert f_close(r) == [price]
    if ok:
        assert issue_codes(r) == [] and r.eligibility is USABLE
    else:
        assert r.eligibility is NOT_USABLE and r.reasons == ("REPRESENTABILITY",)
    check_prefix([bar(day(1), price)])


def test_b25_open_outside_range() -> None:
    r = run([bar(day(1), "10.5", open_="11.2", high="11", low="10")])
    assert r.eligibility is NOT_USABLE and r.reasons == ("REPRESENTABILITY",)
    (i,) = r.issues
    assert i.field == "ohlc"
    check_prefix([bar(day(1), "10.5", open_="11.2", high="11", low="10")])


# ---- B9、B10：可得性与类型、币种 ----


def test_b9_masked_declaration_is_not_prior() -> None:
    bars = [bar(day(1), 50), bar(day(2), 49)]
    dv = [div(day(2), "1", declaration=None, masked="declaration_date|pay_date")]
    r = run(bars, (), dv)
    (rec,) = r.actions
    assert "DECL_NOT_PRIOR" in rec.codes and "DECL_DATE_MISSING" not in rec.codes
    assert r.eligibility is NOT_USABLE and r.reasons == ("DECL_NOT_PRIOR",)
    check_prefix(bars, (), dv)


def test_b10_dividend_types_and_currency() -> None:
    bars = [bar(day(1), 50), bar(day(2), 49)]
    sc = run(bars, (), [div(day(2), "1", dividend_type="SC", currency=None)])
    assert group(sc, group_id_div(day(2), "NONE")).status is GroupStatus.APPLIED
    assert assumption_codes(sc) == ["CURRENCY_ASSUMED_USD"] and sc.eligibility is USABLE
    for t in ("LT", ""):
        r = run(bars, (), [div(day(2), "1", dividend_type=t)])
        (rec,) = r.actions
        assert rec.group_id is None and "DIV_TYPE_UNSUPPORTED" in rec.codes
        assert r.eligibility is NOT_USABLE and r.reasons == ("DIV_TYPE_UNSUPPORTED",)
    usd_not = run(bars, (), [div(day(2), "1", currency="CAD")])
    assert usd_not.reasons == ("CURRENCY_NOT_USD",)
    for t in ("SC", "LT"):
        check_prefix(bars, (), [div(day(2), "1", dividend_type=t, currency=None)])


# ---- B11：ATR 预检 ----


def test_b11_atr_nonpositive_and_single_collapse() -> None:
    flat = [bar(day(i), 10) for i in range(1, 16)]  # 连续 n + 1 = 15 根，F 高 = 低 = 收且不变
    r = run(flat)
    atr = [i for i in r.issues if i.code == "ATR_NONPOSITIVE"]
    assert atr and {i.field for i in atr} == {"ATR14"} and atr[0].day == day(15)
    assert r.eligibility is NOT_USABLE and r.reasons == ("ATR_NONPOSITIVE",)
    # 单日坍缩但有跳空：只计数，ATR 仍为正
    bars = [bar(day(i), 10 + i, high=11 + i, low=9 + i) for i in range(1, 25)]
    bars[20] = bar(day(21), "0.400002", open_="0.400002", high="0.400004", low="0.400001")
    r2 = run(bars, [split(day(21), "10", "1")])
    assert r2.amplitude_collapse_days == (day(21),)
    assert "ATR_NONPOSITIVE" not in issue_codes(r2)
    check_prefix(flat)
    check_prefix(bars, [split(day(21), "10", "1")])


# ---- B12：核实登记对不上 ----


def test_b12_verification_mismatch_raises() -> None:
    bars = [bar(day(1), 50), bar(day(2), 49)]
    dv = [div(day(2), "1")]
    with pytest.raises(AdjustmentInputError, match="对不上"):
        run(
            bars,
            (),
            dv,
            [verification("V1", "group", group_id_div(day(3)), "ABNORMAL_CONFIRMED", "1")],
        )
    with pytest.raises(AdjustmentInputError, match="对不上"):
        run(
            bars,
            (),
            dv,
            [
                verification(
                    "V1",
                    "action",
                    f"{DIV_SOURCE}#7",
                    "DECL_RECORD_CORRECTED",
                    "2015-01-02",
                    DIV_SHA,
                )
            ],
        )
    dup = [div(day(2), "1", 1), div(day(2), "2", 1)]  # 同一来源同一行号：登记对上多条
    with pytest.raises(AdjustmentInputError, match="对上多条"):
        run(
            bars,
            (),
            dup,
            [
                verification(
                    "V1",
                    "action",
                    f"{DIV_SOURCE}#1",
                    "DECL_RECORD_CORRECTED",
                    "2015-01-02",
                    DIV_SHA,
                )
            ],
        )
    with pytest.raises(AdjustmentInputError, match="不一致"):  # 来源哈希不符
        run(
            bars,
            (),
            dv,
            [
                verification(
                    "V1",
                    "action",
                    f"{DIV_SOURCE}#1",
                    "DECL_RECORD_CORRECTED",
                    "2015-01-02",
                    "c" * 64,
                )
            ],
        )
    with pytest.raises(AdjustmentInputError, match="不适用"):  # 类型不适用：该组除权日没有拆股
        run(
            bars, (), dv, [verification("V1", "group", group_id_div(day(2)), "SAME_DAY_BASIS_POST")]
        )
    other = verification(
        "V2", "group", "OTHER|DIV|2015-03-03|PER_SHARE|USD", "ABNORMAL_CONFIRMED", "1"
    )
    assert run(bars, (), dv, [other]).unrelated_verification_ids == ("V2",)  # 不属于本证券：不报错


# ---- B17、B18：合股、拆股与分红的先后 ----


def test_b17_reverse_split_then_dividend() -> None:
    bars = [bar(day(1), 50), bar(day(3), 490)]
    sp, dv = [split(day(2), "10", "1")], [div(day(3), "10")]  # 缺价日 10 并 1；下一有效日除息
    r = run(bars, sp, dv)
    g = group(r, group_id_div(day(3)))
    assert g.c_ref == Decimal(500) and g.b == Decimal("0.98")
    assert r.bars[1].b_cum == Decimal("9.8") and f_close(r)[-1] == "50.000000"
    assert {"DIV_PREV_FALLBACK", "ACTION_DATE_SHIFTED"} <= set(assumption_codes(r))
    check_prefix(bars, sp, dv)


def test_b17_dividend_then_reverse_split() -> None:
    bars = [bar(day(1), 50), bar(day(4), 490)]
    dv, sp = [div(day(2), "1")], [split(day(3), "10", "1")]
    r = run(bars, sp, dv)
    g = group(r, group_id_div(day(2)))
    assert g.c_ref == Decimal(50) and g.b == Decimal("0.98")
    assert r.bars[1].b_cum == Decimal("9.8") and f_close(r)[-1] == "50.000000"
    assert {"DIV_PREV_FALLBACK", "ACTION_DATE_SHIFTED"} <= set(assumption_codes(r))
    check_prefix(bars, sp, dv)


def test_b18_dividend_then_split_shifted_to_same_day() -> None:
    bars = [bar(day(1), 100), bar(day(4), "49.5")]
    dv, sp = [div(day(2), "1")], [split(day(3), "1", "2")]
    r = run(bars, sp, dv)
    g = group(r, group_id_div(day(2)))
    assert g.c_ref == Decimal(100) and g.b == Decimal("0.99")
    assert r.bars[1].b_cum == Decimal("0.495") and f_close(r)[-1] == "100.000000"
    assert group(r, group_id_split(day(3))).effective_day == day(4)
    check_prefix(bars, sp, dv)


# ---- B19、B20：截断视图 ----


def test_b19_truncated_view_has_no_future_conclusions() -> None:
    bars = [bar(D(2015, 3, 2), 50), bar(D(2015, 3, 11), 49)]  # 3 日至 10 日缺价
    dv = [div(D(2015, 3, 3), "1")]
    full = run(bars, (), dv)
    g = group(full, group_id_div(D(2015, 3, 3)))
    assert g.k == 6 and "FALLBACK_TOO_FAR" in g.codes and full.eligibility is NOT_USABLE
    t = D(2015, 3, 3)
    view = view_as_of(full, t)
    (vg,) = view.groups
    assert vg.status is GroupStatus.PENDING and vg.k is None and vg.codes == ()
    assert view.issues == ()
    part = run([bars[0]], (), dv)
    assert view_bytes(view_as_of(part, t)) == view_bytes(view)
    check_prefix(bars, (), dv)


def test_b20_terminal_pending_action() -> None:
    bars = [bar(day(1), 50), bar(day(2), 51)]
    dv = [div(day(5), "1")]  # 最后一根日线之后的原始除权日
    r = run(bars, (), dv)
    g = group(r, group_id_div(day(5)))
    assert g.status is GroupStatus.PENDING and g.effective_day is None
    assert r.terminal_pending == (g.group_id,)
    assert issue_codes(r) == [] and r.eligibility is USABLE
    assert f_close(r) == ["50.000000", "51.000000"]
    check_prefix(bars, (), dv)


# ---- B23：公告日更正后重新判定 ----


@pytest.mark.parametrize(
    ("corrected", "ok"), [(D(2015, 3, 9), True), (D(2015, 3, 10), False), (D(2015, 3, 11), False)]
)
def test_b23_declaration_correction_rejudged(corrected: date, ok: bool) -> None:
    bars = [bar(D(2015, 3, 9), 50), bar(D(2015, 3, 10), 49)]
    dv = [
        dividend_from_text(
            D(2015, 3, 10), "1", "CD", "2015-03-12", "", DIV_SOURCE, 1, DIV_SHA, currency="USD"
        )
    ]
    assert run(bars, (), dv).reasons == ("DECL_NOT_PRIOR",)  # 原始记录：公告日不早于除权日
    v = [
        verification(
            "V5",
            "action",
            f"{DIV_SOURCE}#1",
            "DECL_RECORD_CORRECTED",
            corrected.isoformat(),
            DIV_SHA,
        )
    ]
    r = run(bars, (), dv, v)
    (rec,) = r.actions
    assert rec.verification_ids == ("V5",) and rec.corrected_declaration_date == corrected
    if ok:
        assert "DECL_NOT_PRIOR" not in rec.codes and r.eligibility is USABLE
    else:
        assert "DECL_NOT_PRIOR" in rec.codes and r.eligibility is NOT_USABLE
    check_prefix(bars, (), dv, v)


# ---- B28：区间审计的精确边界（T11R-01） ----

R_PREV, F_PREV = Decimal("1000184977.970371"), Decimal("500092488.985186")
R_DAY, F_DAY = Decimal("2400654331.056531"), Decimal("400109055.176088")
EPS = Decimal("0.0000005")


def _ratio(x: Decimal) -> tuple[int, int]:
    return x.as_integer_ratio()


def _exact(a: Decimal, sign: int, b: Decimal, c: Decimal) -> tuple[int, int]:
    """(a + sign·b)·c 的精确有理数（整数运算，不经过任何有限精度上下文）。"""
    (an, ad), (bn, bd), (cn, cd) = _ratio(a), _ratio(b), _ratio(c)
    return (an * bd + sign * bn * ad) * cn, ad * bd * cd


def test_b28_exact_lower_bound_passes() -> None:
    # 精确下界 (F_prev − ε)·R_day ÷ ((F_day + ε)·R_prev) 恰为 3
    n1, d1 = _exact(F_PREV, -1, EPS, R_DAY)
    n2, d2 = _exact(F_DAY, 1, EPS, R_PREV)
    assert n1 * d2 == 3 * n2 * d1
    assert factor_change_within(R_PREV, F_PREV, R_DAY, F_DAY, (3, 1))


def test_b28_exact_upper_bound_passes_and_below_lower_reported() -> None:
    # 精确上界 (F_prev + ε)·R_day ÷ ((F_day − ε)·R_prev)，以分子分母给出
    n1, d1 = _exact(F_PREV, 1, EPS, R_DAY)
    n2, d2 = _exact(F_DAY, -1, EPS, R_PREV)
    upper = (n1 * d2, d1 * n2)
    assert factor_change_within(R_PREV, F_PREV, R_DAY, F_DAY, upper)
    assert not factor_change_within(R_PREV, F_PREV, R_DAY, F_DAY, (upper[0] + 1, upper[1]))
    below = (3 * 10**28 - 1, 10**28)  # 比精确下界小 10⁻²⁸（精确整数构造）
    assert not factor_change_within(R_PREV, F_PREV, R_DAY, F_DAY, below)
