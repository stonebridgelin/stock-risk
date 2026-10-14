from datetime import date

import pytest

from stock_risk.domain.corrections import (
    CorrectionError,
    CorrectionMethod,
    PriceCorrection,
    apply_corrections,
    parse_method,
)
from stock_risk.domain.listing import ListingPeriod
from stock_risk.domain.market_data import ActionType, CorporateAction, DailyBar, PriceHistory

D = [date(2018, 3, d) for d in (26, 27, 28, 29, 30)]
LISTING = ListingPeriod("SHLD", listed_until=D[3], source_url="https://x")


def bar(d: date, factor: float) -> DailyBar:
    return DailyBar(
        d, 2.5, 2.9, 2.4, 2.8, 100, 2.5 * factor, 2.9 * factor, 2.4 * factor, 2.8 * factor
    )


def history(actions: tuple[CorporateAction, ...] = ()) -> PriceHistory:
    # SHLD 2018-03-28 型：之前系数 1.0，之后约 0.823
    return PriceHistory(
        (bar(D[0], 1.0), bar(D[1], 1.0), bar(D[2], 0.823), bar(D[3], 0.823)), actions
    )


def corr(start: date, end: date) -> PriceCorrection:
    return PriceCorrection("SHLD", start, end, CorrectionMethod.USE_RAW, "PM-03", "依据")


def test_use_raw_replaces_adj_with_raw_inside_range() -> None:
    out = apply_corrections(history(), [corr(D[2], D[3])], LISTING)
    for b in out.bars[2:]:
        assert (b.adj_open, b.adj_high, b.adj_low, b.adj_close) == (b.open, b.high, b.low, b.close)


def test_bars_outside_range_unchanged() -> None:
    h = history()
    out = apply_corrections(h, [corr(D[2], D[2])], LISTING)
    assert out.bars[:2] == h.bars[:2] and out.bars[3] == h.bars[3]
    assert out.bars[2].adj_close == out.bars[2].close


def test_unknown_method_rejected() -> None:
    with pytest.raises(CorrectionError):
        parse_method("scale")
    with pytest.raises(CorrectionError):
        PriceCorrection("SHLD", D[2], D[3], "scale", "PM-03", "依据")  # type: ignore[arg-type]


def test_action_inside_range_rejected() -> None:
    h = history((CorporateAction(D[3], ActionType.DIVIDEND, 0.1),))
    with pytest.raises(CorrectionError):
        apply_corrections(h, [corr(D[2], D[3])], LISTING)


def test_action_outside_range_allowed() -> None:
    h = history((CorporateAction(D[0], ActionType.DIVIDEND, 0.1),))
    out = apply_corrections(h, [corr(D[2], D[3])], LISTING)
    assert out.actions == h.actions


def test_range_must_be_inside_listing_period() -> None:
    with pytest.raises(CorrectionError):
        apply_corrections(history(), [corr(D[2], D[4])], LISTING)


def test_ruling_and_evidence_required() -> None:
    with pytest.raises(CorrectionError):
        PriceCorrection("SHLD", D[2], D[3], CorrectionMethod.USE_RAW, "", "依据")
    with pytest.raises(CorrectionError):
        PriceCorrection("SHLD", D[3], D[2], CorrectionMethod.USE_RAW, "PM-03", "依据")
