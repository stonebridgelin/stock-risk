"""T09-E3：18 号 C23、C23m（波动分组）。期望值取自夹具 JSON；验证期新证券的期望按 17 §3.1 推导。"""

from decimal import Decimal

from stock_risk.domain.evaluation.vol_groups import (
    VolGroup,
    assign_group,
    boundaries,
    daily_vol_ratios,
    median,
    new_security_stat,
    security_stat,
)
from tests.unit.evaluation.helpers import load_fixtures, make_series


def c_case(case_id: str) -> dict:  # type: ignore[type-arg]
    return load_fixtures()["C"][case_id]


def test_C23_type7_tertiles_and_groups() -> None:
    case = c_case("C23")
    bounds = boundaries([Decimal(x) for x in case["selection_stats"]])
    assert bounds.b1 == Decimal(case["expected_b1"])
    assert bounds.b2 == Decimal(case["expected_b2"])
    for stat, group in {**case["expected_groups"], **case["boundary_cases"]}.items():
        assert assign_group(Decimal(stat), bounds) is VolGroup(group), stat


def test_C23_validation_new_security() -> None:
    """日比率取 i / 1000（i = 1, 2, …），所用前 m 个的中位数可直接写出。"""
    case = c_case("C23")
    by_days = {v["evaluable_days"]: v["expected"] for v in case["validation_new"]}
    assert set(by_days) == {62, 63, 251, 300}

    def ratios(n: int) -> list[Decimal]:
        return [Decimal(i) / Decimal(1000) for i in range(1, n + 1)]

    assert new_security_stat(ratios(62)) is None  # 未归组（UNGROUPED）
    assert assign_group(None, boundaries([Decimal("0.01")])) is VolGroup.UNGROUPED
    assert new_security_stat(ratios(63)) == Decimal("0.032")  # 全部 63 日：第 32 个
    assert new_security_stat(ratios(251)) == Decimal("0.126")  # 全部 251 日：第 126 个
    assert new_security_stat(ratios(300)) == Decimal("0.1265")  # 前 252 日：(126 + 127) / 2


def test_C23m_even_median() -> None:
    case = c_case("C23m")
    stat = security_stat([Decimal(x) for x in case["ratios"]])
    assert stat == Decimal(case["expected_median"])
    assert str(stat) == case["expected_median"]  # 量化到 12 位


def test_daily_vol_ratios_only_mask_days() -> None:
    series = make_series(c_case("C42c"))  # 新上市：第 82 日起进入掩码
    ratios = daily_vol_ratios(series)
    assert len(ratios) == sum(series.mask) == len(series.calendar.days) - 82
    assert median(list(ratios)) == security_stat(ratios)
