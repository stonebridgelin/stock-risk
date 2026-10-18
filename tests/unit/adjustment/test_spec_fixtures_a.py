"""规格夹具 A1–A21（D0a 复权规格 v4 第 12 节；T11-D0a 第 9.1 节）。

期望值照抄规格第 12 节，不由实现输出回填。除数值外，每个夹具还断言日志状态、假设与问题代码、资格。
每个夹具另做前缀不变性检查（第 9.3 节）。
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, localcontext

from stock_risk.domain.adjustment import view_as_of
from stock_risk.domain.adjustment.model import Eligibility, GroupStatus
from stock_risk.domain.audit import CanonicalBarPrices, canonical_factor_checks
from stock_risk.domain.corrections import parse_raw_price_correction
from stock_risk.domain.numeric import PARSE_CONTEXT
from tests.unit.adjustment.helpers import (
    SEC,
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
APPLIED, ANCHOR, PENDING, REJECTED = (
    GroupStatus.APPLIED,
    GroupStatus.ANCHOR_NOT_APPLIED,
    GroupStatus.PENDING,
    GroupStatus.REJECTED,
)


def test_a1_single_split() -> None:
    bars = [
        bar(day(1), 100, volume=1000),
        bar(day(2), 51, volume=2200),
        bar(day(3), 52, volume=2000),
    ]
    sp = [split(day(2), "1", "2")]
    r = run(bars, sp)
    assert [b.b_cum for b in r.bars] == [Decimal(1), Decimal("0.5"), Decimal("0.5")]
    assert f_close(r) == ["100.000000", "102.000000", "104.000000"]
    assert [str(b.f_volume) for b in r.bars] == ["1000.000000", "1100.000000", "1000.000000"]
    for b in r.bars:  # 成交额不变：F × F_V = R × V
        assert b.f_close * b.f_volume == b.close * b.volume
    assert group(r, group_id_split(day(2))).status is APPLIED
    assert assumption_codes(r) == ["DECL_DATE_MISSING"]  # 拆股默认（第 9.1 节）
    assert issue_codes(r) == [] and r.eligibility is USABLE
    check_prefix(bars, sp)


def test_a2_single_dividend() -> None:
    bars = [bar(day(1), 50), bar(day(2), "49.5")]
    dv = [div(day(2), "1.00")]
    r = run(bars, (), dv)
    g = group(r, group_id_div(day(2)))
    assert g.b == Decimal("0.98") and g.status is APPLIED
    assert f_close(r) == ["50.000000", "50.510204"]
    assert assumption_codes(r) == [] and issue_codes(r) == [] and r.eligibility is USABLE
    check_prefix(bars, (), dv)


def _a3_inputs() -> tuple[list, list, list]:
    bars = [bar(day(1), 100), bar(day(2), "49.0")]
    return bars, [split(day(2), "1", "2")], [div(day(2), "0.50")]


def test_a3_same_day_split_and_dividend_verified_post() -> None:
    bars, sp, dv = _a3_inputs()
    v = [verification("V1", "group", group_id_div(day(2)), "SAME_DAY_BASIS_POST")]
    r = run(bars, sp, dv, v)
    g = group(r, group_id_div(day(2)))
    assert g.c_ref == Decimal(50) and g.b == Decimal("0.99")
    assert g.status is APPLIED and g.verification_ids == ("V1",)
    assert r.bars[1].b_cum == Decimal("0.495")
    assert f_close(r) == ["100.000000", "98.989899"]
    assert assumption_codes(r) == ["DECL_DATE_MISSING"]
    assert issue_codes(r) == [] and r.eligibility is USABLE
    check_prefix(bars, sp, dv, v)


def test_a4_special_dividend_counted() -> None:
    # 裁决 A-2：SC 计入，结果同 A2
    bars = [bar(day(1), 50), bar(day(2), "49.5")]
    dv = [div(day(2), "1.00", dividend_type="SC")]
    r = run(bars, (), dv)
    assert group(r, group_id_div(day(2))).b == Decimal("0.98")
    assert f_close(r) == ["50.000000", "50.510204"]
    assert issue_codes(r) == [] and r.eligibility is USABLE
    check_prefix(bars, (), dv)


def test_a5_missing_day_before_ex_date() -> None:
    bars = [bar(day(1), 50), bar(day(3), 48)]  # 第 2 日缺价
    dv = [div(day(3), "1.00")]
    r = run(bars, (), dv)
    g = group(r, group_id_div(day(3)))
    assert g.c_ref == Decimal(50) and g.b == Decimal("0.98") and g.k == 1
    assert f_close(r) == ["50.000000", "48.979592"]
    assert assumption_codes(r) == ["DIV_PREV_FALLBACK"]
    assert issue_codes(r) == [] and r.eligibility is USABLE
    check_prefix(bars, (), dv)


def test_a6_anchor_day_action_not_applied() -> None:
    bars = [bar(day(1), 50), bar(day(2), 51)]
    sp = [split(day(1), "1", "2")]  # 锚日为 1 拆 2 的除权日
    r = run(bars, sp)
    assert f_close(r) == ["50.000000", "51.000000"]
    g = group(r, group_id_split(day(1)))
    assert g.status is ANCHOR and g.b is None
    # 锚日行动不作用于 F，不依赖其可得性假设，原始记录不带标记（见回执"偏离任务卡之处"）
    assert assumption_codes(r) == [] and issue_codes(r) == [] and r.eligibility is USABLE
    check_prefix(bars, sp)


def test_a7_relation_to_s_of_t() -> None:
    bars = [bar(day(1), 100), bar(day(2), 51), bar(day(3), 52)]
    r = run(bars, [split(day(2), "1", "2")])
    b_t = r.bars[2].b_cum  # T = 第 3 日
    assert [b.f_close * b_t for b in r.bars] == [Decimal(50), Decimal(51), Decimal(52)]


def test_a8_delisted_before_2019() -> None:
    bars = [bar(date(2014, 6, 2), 40), bar(date(2014, 6, 9), 21), bar(date(2015, 3, 31), 18)]
    sp = [split(date(2014, 6, 9), "1", "2")]
    r = run(bars, sp)
    assert f_close(r) == ["40.000000", "42.000000", "36.000000"]
    assert max(g.original_ex_date for g in r.groups) <= date(2015, 3, 31)
    assert r.eligibility is USABLE
    check_prefix(bars, sp)


def test_a9_reverse_split() -> None:
    bars = [bar(day(1), "1.00", volume=50000), bar(day(2), "10.20", volume=5200)]
    sp = [split(day(2), "10", "1")]
    r = run(bars, sp)
    assert group(r, group_id_split(day(2))).b == Decimal(10)
    assert f_close(r) == ["1.000000", "1.020000"]
    assert [str(b.f_volume) for b in r.bars] == ["50000.000000", "52000.000000"]
    assert r.eligibility is USABLE
    check_prefix(bars, sp)


def test_a10_abnormal_dividend() -> None:
    bars = [bar(day(1), 10), bar(day(2), 4)]
    dv = [div(day(2), "6.00")]
    r = run(bars, (), dv)
    g = group(r, group_id_div(day(2)))
    assert g.b == Decimal("0.4") and g.status is REJECTED
    assert issue_codes(r) == ["ABNORMAL_FACTOR"]
    assert r.eligibility is NOT_USABLE and r.reasons == ("ABNORMAL_FACTOR",)
    check_prefix(bars, (), dv)


def _a11_inputs() -> tuple[list, list, list]:
    bars = [
        bar(day(1), 100),
        bar(day(2), 50),
        bar(day(3), 59, open_=49, high=49, low=49),  # 原始收盘有误（59），应为 49
    ]
    corr = [
        parse_raw_price_correction(
            {
                "id": "C1",
                "security": SEC,
                "day": day(3).isoformat(),
                "field": "close",
                "value": "49",
                "ruling": "PM-T",
                "evidence": "测试",
                "version": "1",
            }
        )
    ]
    return bars, [split(day(2), "1", "2")], corr


def test_a11_raw_price_correction_before_adjustment() -> None:
    bars, sp, corr = _a11_inputs()
    r = run(bars, sp, corrections=corr)
    assert f_close(r) == ["100.000000", "100.000000", "98.000000"]
    assert r.raw_correction_ids == ("C1",) and r.eligibility is USABLE
    check_prefix(bars, sp, corrections=corr)


def test_a12_two_dividends_same_day_summed() -> None:
    bars = [bar(day(1), 100), bar(day(2), 97)]
    dv = [div(day(2), "1", 1), div(day(2), "2", 2)]
    r = run(bars, (), dv)
    (g,) = r.groups
    assert len(g.members) == 2 and g.dividend_total == Decimal(3)
    assert g.b == Decimal("0.97")
    assert f_close(r) == ["100.000000", "100.000000"]  # 逐笔相乘会得 99.979386
    assert r.eligibility is USABLE
    check_prefix(bars, (), dv)


def test_a13_split_in_missing_gap_then_dividend() -> None:
    bars = [bar(day(1), 50), bar(day(3), 24)]  # 第 2 日 1 拆 2 且缺价
    sp, dv = [split(day(2), "1", "2")], [div(day(3), "1")]
    r = run(bars, sp, dv)
    gs, gd = group(r, group_id_split(day(2))), group(r, group_id_div(day(3)))
    assert gs.effective_day == day(3) and "ACTION_DATE_SHIFTED" in gs.codes
    assert gd.c_ref == Decimal(25) and gd.b == Decimal("0.96")
    assert r.bars[1].b_cum == Decimal("0.48")
    assert f_close(r) == ["50.000000", "50.000000"]  # 不换算会得 48.979592
    assert "SAME_DAY_BASIS_UNVERIFIED" not in issue_codes(r)
    assert assumption_codes(r) == ["ACTION_DATE_SHIFTED", "DECL_DATE_MISSING", "DIV_PREV_FALLBACK"]
    assert r.eligibility is USABLE
    check_prefix(bars, sp, dv)


def test_a14_audit_interval_with_missing_ex_date() -> None:
    d3, d4, d5 = date(2017, 1, 3), date(2017, 1, 4), date(2017, 1, 5)
    cal = (d3, d4, d5)
    r = run([bar(d3, 100), bar(d5, 50)], [split(d4, "1", "2")], calendar=cal)
    assert f_close(r) == ["100.000000", "100.000000"]
    prices = [CanonicalBarPrices(b.day, b.close, b.f_close) for b in r.bars]
    assert canonical_factor_checks(prices, r.groups) == []  # 预期因子变化 0.5，不报告跳变


def test_a15_representability() -> None:
    # 累计拆股因子 B = 10⁶（六次 10 并 1），原始收盘 0.40：F = 0.0000004，量化后为 0
    bars = [bar(day(1), 1), bar(day(2), "0.40")]
    sp = [split(day(2), "10", "1", row) for row in range(1, 7)]
    r = run(bars, sp)  # 先检查后计算，不让运行失败
    assert r.bars[1].b_cum == Decimal(10) ** 6
    assert r.bars[1].f_close == Decimal("0.000000")
    assert "REPRESENTABILITY" in issue_codes(r)
    assert r.eligibility is NOT_USABLE and r.reasons == ("REPRESENTABILITY",)


def test_a16_same_day_basis_unverified() -> None:
    bars, sp, dv = _a3_inputs()
    r = run(bars, sp, dv)
    assert group(r, group_id_div(day(2))).status is REJECTED
    assert issue_codes(r) == ["SAME_DAY_BASIS_UNVERIFIED"]
    assert r.eligibility is NOT_USABLE
    check_prefix(bars, sp, dv)


def test_a17_shifted_onto_anchor() -> None:
    bars = [bar(day(1), 100), bar(day(2), 99)]  # 锚日 2015-03-02（周一）
    dv = [div(date(2015, 2, 28), "1")]  # 前一个周六
    r = run(bars, (), dv)
    g = group(r, group_id_div(date(2015, 2, 28)))
    assert g.status is ANCHOR and g.effective_day == day(1)
    assert g.original_ex_date == date(2015, 2, 28)
    assert f_close(r) == ["100.000000", "99.000000"]
    assert r.eligibility is USABLE
    check_prefix(bars, (), dv)


def _a18_bars() -> list:
    return [
        bar(day(1), "0.8"),
        bar(day(2), "0.400002", open_="0.400002", high="0.400004", low="0.400001"),
    ]


def test_a18_amplitude_collapse() -> None:
    r2 = run(_a18_bars(), [split(day(2), "2", "1")])  # B = 2
    b = r2.bars[1]
    assert (str(b.f_high), str(b.f_low)) == ("0.200002", "0.200000")  # 低价恰在中点，取偶数
    assert r2.amplitude_collapse_days == () and issue_codes(r2) == []  # 该日价格检查通过
    r10 = run(_a18_bars(), [split(day(2), "10", "1")])  # B = 10
    b = r10.bars[1]
    assert (str(b.f_high), str(b.f_low)) == ("0.040000", "0.040000")
    assert r10.amplitude_collapse_days == (day(2),)  # 只计数
    assert "REPRESENTABILITY" not in issue_codes(r10)


def test_a19_two_dividends_in_missing_gap() -> None:
    bars = [bar(day(1), 100), bar(day(4), 97)]  # 第 2、3 日缺价
    dv = [div(day(2), "1", 1), div(day(3), "2", 2)]
    r = run(bars, (), dv)
    g1, g2 = group(r, group_id_div(day(2))), group(r, group_id_div(day(3)))
    assert g1.effective_day == g2.effective_day == day(4)
    assert g1.b == Decimal("0.99") and g2.c_ref == Decimal(99)
    with localcontext(PARSE_CONTEXT):
        assert g2.b == Decimal(97) / Decimal(99)
        assert r.bars[1].b_cum.quantize(Decimal("1e-20")) == Decimal("0.97")
    assert f_close(r) == ["100.000000", "100.000000"]  # 逐笔都用 100 得 0.9702
    for g in (g1, g2):
        assert set(g.codes) == {"DIV_PREV_FALLBACK", "ACTION_DATE_SHIFTED"}
    assert r.eligibility is USABLE
    check_prefix(bars, (), dv)


def test_a20_same_day_basis_pre() -> None:
    bars = [bar(day(1), 100), bar(day(2), "49.5")]
    sp = [split(day(2), "1", "2")]
    pre = run(
        bars,
        sp,
        [div(day(2), "1")],
        [verification("V1", "group", group_id_div(day(2)), "SAME_DAY_BASIS_PRE")],
    )
    g = group(pre, group_id_div(day(2)))
    assert g.c_ref == Decimal(50) and g.b == Decimal("0.99")
    assert pre.bars[1].b_cum == Decimal("0.495")
    assert f_close(pre) == ["100.000000", "100.000000"]
    post = run(
        bars,
        sp,
        [div(day(2), "0.5")],
        [verification("V1", "group", group_id_div(day(2)), "SAME_DAY_BASIS_POST")],
    )
    assert f_close(post) == f_close(pre)  # 与按新股每股 0.5 元登记的结果相同
    assert pre.eligibility is USABLE and post.eligibility is USABLE


def test_a21_truncated_on_pending_action() -> None:
    bars = [bar(day(1), 100), bar(day(3), 99)]  # 第 2 日缺价且为除权日
    dv = [div(day(2), "1")]
    full = run(bars, (), dv)
    g = group(full, group_id_div(day(2)))
    assert g.effective_day == day(3) and g.b == Decimal("0.99")
    t = day(2)
    view = view_as_of(full, t)
    (vg,) = view.groups
    assert vg.status is PENDING and vg.effective_day is None and vg.b is None and vg.k is None
    assert [str(b.f_close) for b in view.bars] == ["100.000000"]
    part = run([bars[0]], (), dv)
    assert view_bytes(view_as_of(part, t)) == view_bytes(view)
    check_prefix(bars, (), dv)
