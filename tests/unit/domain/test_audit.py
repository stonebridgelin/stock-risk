import math
from datetime import date, timedelta

from stock_risk.domain.audit import (
    FailureKind,
    PriceBasis,
    adj_factor_jumps,
    calendar_gaps,
    count_sealed_rows,
    explicit_failures,
    zero_volume_days,
)
from stock_risk.domain.market_data import DailyBar

D0 = date(2018, 3, 26)


def day(i: int) -> date:
    return D0 + timedelta(days=i)


def bar(i: int, o=10.0, h=11.0, lo=9.0, c=10.0, v=100.0, factor=1.0, adj=None) -> DailyBar:
    a = adj if adj is not None else (o * factor, h * factor, lo * factor, c * factor)
    return DailyBar(day(i), o, h, lo, c, v, *a)


def kinds(bars) -> list[tuple[PriceBasis | None, FailureKind]]:
    return [(f.basis, f.kind) for f in explicit_failures(bars)]


def test_clean_bar_has_no_failure() -> None:
    assert explicit_failures([bar(0)]) == []


def test_high_below_low_raw_only() -> None:
    b = bar(0, h=8.0, lo=9.0, c=8.5, adj=(10, 11, 9, 10))
    assert (PriceBasis.RAW, FailureKind.HIGH_BELOW_LOW) in kinds([b])
    assert all(basis is PriceBasis.RAW for basis, _ in kinds([b]))


def test_close_outside_range_checked_on_adj_separately() -> None:
    b = bar(0, adj=(10, 11, 9, 11.5))
    assert kinds([b]) == [(PriceBasis.ADJ, FailureKind.CLOSE_OUTSIDE_RANGE)]


def test_close_on_boundary_is_ok() -> None:
    assert explicit_failures([bar(0, c=11.0), bar(1, c=9.0)]) == []


def test_non_positive_price() -> None:
    assert kinds([bar(0, o=0.0, adj=(10, 11, 9, 10))]) == [
        (PriceBasis.RAW, FailureKind.NON_POSITIVE_PRICE)
    ]


def test_non_finite_price_and_volume() -> None:
    k = kinds([bar(0, c=math.nan), bar(1, v=math.inf)])
    assert (PriceBasis.RAW, FailureKind.NON_FINITE) in k
    assert (None, FailureKind.NON_FINITE) in k


def test_adj_factor_jump_without_action_reported() -> None:
    # SHLD 2018-03-28 型：系数 1.0 → 0.823，当天无公司行动
    bars = [bar(0), bar(1), bar(2, factor=0.823)]
    (j,) = adj_factor_jumps(bars, set())
    assert j.day == day(2) and j.prev_day == day(1)
    assert abs(j.change - (-0.177)) < 1e-9


def test_adj_factor_jump_with_action_or_small_change_ignored() -> None:
    bars = [bar(0), bar(1, factor=0.9), bar(2, factor=0.9 * 1.019)]
    assert adj_factor_jumps(bars, {day(1)}) == []


def test_adj_factor_jump_threshold_is_strict() -> None:
    bars = [bar(0, factor=1.0), bar(1, factor=0.98)]
    assert adj_factor_jumps(bars, set(), threshold=0.0200001) == []


def test_calendar_gaps_between_first_and_last_bar() -> None:
    cal = [day(i) for i in range(6)]
    bars = [bar(1), bar(2), bar(4)]
    assert calendar_gaps(bars, cal) == [day(3)]


def test_calendar_gaps_bounded_by_listing_period() -> None:
    cal = [day(i) for i in range(8)]
    bars = [bar(2), bar(3), bar(5), bar(6), bar(7)]
    # 上市期间 day1 ~ day5：上市起首日前缺价、退市日后不计
    assert calendar_gaps(bars, cal, start=day(1), end=day(5)) == [day(1), day(4)]


def test_calendar_gaps_empty_bars() -> None:
    assert calendar_gaps([], [day(0)]) == []


def test_zero_volume_and_sealed_count() -> None:
    assert zero_volume_days([bar(0, v=0.0), bar(1)]) == [day(0)]
    assert count_sealed_rows([date(2019, 12, 31), date(2020, 1, 2), date(2020, 1, 3)]) == 2


def test_manifest_mismatches() -> None:
    from stock_risk.domain.audit import ManifestStatus, manifest_mismatches

    expected = {"A.csv": "aa", "B.csv": "bb", "C.csv": "cc"}
    actual = {"A.csv": "AA", "B.csv": "xx", "D.csv": "dd"}
    out = {m.file: m.status for m in manifest_mismatches(expected, actual)}
    assert out == {
        "B.csv": ManifestStatus.HASH_MISMATCH,
        "C.csv": ManifestStatus.FILE_MISSING,
        "D.csv": ManifestStatus.NOT_IN_MANIFEST,
    }


def test_corrected_shld_type_jump_disappears() -> None:
    # T00a 验收：更正后对复权系数跳变的检查不再出现 2018-03-28 型跳变
    from stock_risk.domain.corrections import CorrectionMethod, PriceCorrection, apply_corrections
    from stock_risk.domain.market_data import PriceHistory

    bars = (bar(0), bar(1), bar(2, factor=0.823), bar(3, factor=0.823))
    assert [j.day for j in adj_factor_jumps(bars, set())] == [day(2)]
    c = PriceCorrection("SHLD", day(2), day(3), CorrectionMethod.USE_RAW, "PM-03", "依据")
    corrected = apply_corrections(PriceHistory(bars, ()), [c])
    assert adj_factor_jumps(corrected.bars, set()) == []
