"""T09 E2：占用率、F 与通知负担的单元测试（期望值自行推导，推导写在注释中，并列入回执）。

样本：18 号 B13 的价格（195 日，无缺价；掩码自第 82 日起，第 82–194 日共 113 个掩码日）与
B13 的方法 M_N_RESTART：在警区段 [101, 103]、[106, 115]（共 3 + 10 = 13 日，全部在掩码内）；
事件 START@101、RELEASE@104、START@106、RELEASE@116。
"""

from decimal import Decimal

from stock_risk.domain.evaluation.membership import (
    Membership,
    Rate,
    notice_stats,
    occupancy,
    summarize_rates,
)
from tests.unit.evaluation.e2_helpers import b_case, methods
from tests.unit.evaluation.helpers import run_case


def _sample():  # type: ignore[no-untyped-def]
    case = b_case("B13")
    series, _ = run_case(case)
    return series, methods(series, case)["M_N_RESTART"]


def test_whole_period_occupancy_f_and_burden() -> None:
    series, m = _sample()
    assert sum(series.mask) == 113  # 第 82–194 日
    occ = occupancy(series, m, None)
    # 13 / 113 = 0.115044247787 61… → 12 位 HALF_EVEN：0.115044247788
    assert (occ.numerator, occ.denominator, occ.value) == (13, 113, Decimal("0.115044247788"))
    stats = notice_stats(series, m, None)
    assert stats.days == 113
    assert stats.counts["START"] == 2 and stats.counts["RELEASE"] == 2
    assert stats.counts["ESCALATE"] == 0 and stats.counts["DOWNGRADE"] == 0
    # F = (START 2 + ESCALATE 0) × 252 / 113 = 504 / 113 = 4.460176991150 44… → 4.460176991150
    assert stats.f.value == Decimal("4.460176991150")
    # RELEASE 每证券年同为 2 × 252 / 113
    assert stats.per_year["RELEASE"].value == Decimal("4.460176991150")
    assert stats.per_year["DOWNGRADE"].value == Decimal("0")


def test_membership_restricts_days_and_events() -> None:
    series, m = _sample()
    days = series.calendar.days
    member = Membership(((days[101], days[110]),))  # [101, 110)：掩码日 101–109 共 9 日
    occ = occupancy(series, m, member)
    # 在警：101、102、103、106、107、108、109 共 7 日 → 7 / 9 = 0.777777777777 7… → 0.777777777778
    assert (occ.numerator, occ.denominator, occ.value) == (7, 9, Decimal("0.777777777778"))
    stats = notice_stats(series, m, member)
    # 期间内的事件：START@101、RELEASE@104、START@106（RELEASE@116 在期间外）
    assert stats.counts["START"] == 2 and stats.counts["RELEASE"] == 1
    assert stats.f.value == Decimal("56")  # 2 × 252 / 9
    assert stats.per_year["RELEASE"].value == Decimal("28")  # 1 × 252 / 9


def test_zero_days_gives_empty_rate_with_reason_I13() -> None:
    """I-13：分母为 0 时比率为空、带原因代码，汇总时单独计数而不是静默丢弃。"""
    series, m = _sample()
    days = series.calendar.days
    before_mask = Membership(((days[0], days[50]),))  # 掩码自第 82 日起：期间内没有掩码日
    occ = occupancy(series, m, before_mask)
    assert occ == Rate(None, 0, 0, "NO_EVALUABLE_DAYS")
    stats = notice_stats(series, m, before_mask)
    assert stats.days == 0 and stats.f.value is None and stats.f.reason == "NO_EVALUABLE_DAYS"
    assert all(r.value is None for r in stats.per_year.values())
    summary = summarize_rates([occupancy(series, m, None), occ, stats.f])
    assert summary.defined == (Decimal("0.115044247788"),)
    assert summary.undefined == 2 and summary.reasons == {"NO_EVALUABLE_DAYS": 2}
