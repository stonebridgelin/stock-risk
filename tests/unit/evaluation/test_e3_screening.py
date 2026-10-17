"""T09-E3：18 号 D28a–c、D38（季度筛选）。期望值取自夹具 JSON。

v1.3 起 W1 有基础成交量（T09-E3M），四个夹具完整比较。
"""

from decimal import Decimal
from typing import Any

import pytest

from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.evaluation.screening import (
    Eligibility,
    ScreenBar,
    ScreenSecurity,
    screen_quarter,
)
from stock_risk.domain.numeric import InvalidInputError
from tests.unit.evaluation.helpers import load_fixtures, workdays

G, M_MASK = 10, 83  # 18 号约定


def d_case(case_id: str) -> dict[str, Any]:
    return load_fixtures()["D"][case_id]


def volume_of(raw_volume: int | dict[str, int], day: int) -> Decimal:
    """conventions.raw_volume：整数为每日相同；对象的 "default" 为其余日子，其他键为日序号。"""
    if isinstance(raw_volume, int):
        return Decimal(raw_volume)
    if "default" not in raw_volume:
        raise ValueError("raw_volume 的对象写法必须给出 default")
    return Decimal(raw_volume.get(str(day), raw_volume["default"]))


def make_security(spec: dict[str, Any], days: tuple) -> ScreenSecurity:  # type: ignore[type-arg]
    bars = tuple(
        ScreenBar(days[i], Decimal(spec["raw_close"]), volume_of(spec["raw_volume"], i))
        for lo, hi in spec["bar_days"]
        for i in range(lo, hi + 1)
    )
    return ScreenSecurity(spec["id"], spec["issuer"], spec["class_ok"], bars)


def assert_screen(case_id: str, securities_of: str | None = None) -> None:
    case = d_case(case_id)
    specs = d_case(securities_of or case_id)["securities"]
    last = max([case["screen_day"], *(hi for s in specs for _, hi in s["bar_days"])])
    days = workdays(last + 1)
    result = screen_quarter(
        TradingCalendar(days),
        days[case["screen_day"]],
        [make_security(s, days) for s in specs],
        case["N_u"],
        set(case["exclude_issuers"]),
        case["rep_map_before"],
        G,
        M_MASK,
    )
    expected = case["expected"]
    assert {k: v.value for k, v in result.eligibility.items()} == expected["eligibility"]
    assert list(result.ranking) == [(c, Decimal(v)) for c, v in expected["ranking"]]
    assert list(result.selected) == expected["selected"]
    assert dict(result.rep_map) == expected["rep_map_after"]


def test_D28a_development_screen() -> None:
    assert_screen("D28a")


def test_D28b_fixed_rep_kept_and_first_appearance() -> None:
    assert_screen("D28b")


def test_D28c_rep_suspended_issuer_absent() -> None:
    assert_screen("D28c")


def test_D38_validation_screen_keeps_exploration_issuers() -> None:
    assert_screen("D38", securities_of="D28a")


def test_raw_volume_forms() -> None:
    assert volume_of(900, 5) == Decimal(900)
    assert volume_of({"default": 700, "100": 1000000}, 99) == Decimal(700)
    assert volume_of({"default": 700, "100": 1000000}, 100) == Decimal(1000000)
    with pytest.raises(ValueError, match="default"):
        volume_of({"100": 1000000}, 99)


def test_screen_reason_order_and_inputs() -> None:
    days = workdays(120)
    cal = TradingCalendar(days)
    bars = tuple(ScreenBar(days[i], Decimal("4"), Decimal(0)) for i in range(0, 100))
    # 类别不合格优先；价格与成交额都不满足时记第一个（⑤）
    secs = [ScreenSecurity("A1", "A", False, bars), ScreenSecurity("B1", "B", True, bars)]
    result = screen_quarter(cal, days[100], secs, 1, set(), {}, G, M_MASK)
    assert result.eligibility == {"A1": Eligibility.CLASS, "B1": Eligibility.PRICE_LT_5}
    assert result.selected == () and dict(result.rep_map) == {}
    with pytest.raises(InvalidInputError):
        screen_quarter(cal, days[100], secs, 0, set(), {}, G, M_MASK)
    with pytest.raises(InvalidInputError):
        screen_quarter(cal, days[0], secs, 1, set(), {}, G, M_MASK)


def test_screen_trailing_break_is_warmup() -> None:
    """G = 2 时 3 日尾部缺价已构成断点，前一交易日不在任何历史段内。"""
    days = workdays(120)
    bars = tuple(ScreenBar(days[i], Decimal(20), Decimal(100)) for i in range(0, 97))
    result = screen_quarter(
        TradingCalendar(days),
        days[100],
        [ScreenSecurity("A1", "A", True, bars)],
        1,
        set(),
        {},
        2,
        M_MASK,
    )
    assert result.eligibility["A1"] is Eligibility.WARMUP
