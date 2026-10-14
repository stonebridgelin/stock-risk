from datetime import date

import pytest

from stock_risk.domain.market_data import ActionType, CorporateAction, DailyBar, PriceHistory
from stock_risk.domain.segments import (
    SegmentError,
    SymbolSegment,
    junction_checks,
    stitch,
    validate_segments,
)


def bar(d: date, o: float = 10, c: float = 10, factor: float = 1.0) -> DailyBar:
    return DailyBar(
        d,
        o,
        max(o, c) + 1,
        min(o, c) - 1,
        c,
        100,
        o * factor,
        (max(o, c) + 1) * factor,
        (min(o, c) - 1) * factor,
        c * factor,
    )


D1, D2, D3, D4 = date(2018, 7, 12), date(2018, 7, 13), date(2018, 7, 16), date(2018, 7, 17)
OLD = SymbolSegment("VRX", "VRX", None, D2)
NEW = SymbolSegment("VRX", "BHC", D3, None)


def test_segment_rejects_reversed_range() -> None:
    with pytest.raises(SegmentError):
        SymbolSegment("VRX", "VRX", D3, D2)


def test_validate_rejects_overlap_and_unsorted() -> None:
    with pytest.raises(SegmentError):
        validate_segments([SymbolSegment("VRX", "VRX", None, D3), NEW])
    with pytest.raises(SegmentError):
        validate_segments([NEW, OLD])


def test_validate_rejects_open_inner_boundary_and_mixed_symbols() -> None:
    with pytest.raises(SegmentError):
        validate_segments([SymbolSegment("VRX", "VRX"), NEW])
    with pytest.raises(SegmentError):
        validate_segments([OLD, SymbolSegment("BHC", "BHC", D3)])
    with pytest.raises(SegmentError):
        validate_segments([])


def test_stitch_clips_each_segment_and_sorts() -> None:
    # 两个数据源各自多返回了对方区间的日线，拼接时按分段截取
    old = PriceHistory(
        (bar(D1), bar(D2), bar(D3)), (CorporateAction(D3, ActionType.DIVIDEND, 1.0),)
    )
    new = PriceHistory(
        (bar(D2), bar(D3), bar(D4)), (CorporateAction(D2, ActionType.DIVIDEND, 2.0),)
    )
    h = stitch([(OLD, old), (NEW, new)])
    assert [b.day for b in h.bars] == [D1, D2, D3, D4]
    assert h.actions == ()


def test_stitch_keeps_actions_inside_segment() -> None:
    old = PriceHistory((bar(D1),), (CorporateAction(D1, ActionType.SPLIT, 2.0),))
    new = PriceHistory((bar(D3),), ())
    h = stitch([(OLD, old), (NEW, new)])
    assert [(a.day, a.value) for a in h.actions] == [(D1, 2.0)]


def test_junction_checks_ratio_and_factor_change() -> None:
    bars = [bar(D1), bar(D2, c=20, factor=0.5), bar(D3, o=21, factor=0.6), bar(D4)]
    (j,) = junction_checks(bars, [OLD, NEW])
    assert (j.prev_day, j.next_day) == (D2, D3)
    assert j.prev_close == 20 and j.next_open == 21
    assert j.open_to_close_ratio == pytest.approx(1.05)
    assert j.adj_factor_change == pytest.approx(0.6 / 0.5 - 1)


def test_junction_checks_reports_missing_side() -> None:
    (j,) = junction_checks([bar(D1), bar(D2)], [OLD, NEW])
    assert j.next_day is None and j.open_to_close_ratio is None
