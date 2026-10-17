"""D-03、D-05–D-07：ATR、短缺价、断点与公共掩码。"""

from dataclasses import replace
from decimal import Decimal, localcontext

import pytest

from stock_risk.domain.evaluation.event_model import EventParameters
from stock_risk.domain.evaluation.events import build_events
from stock_risk.domain.evaluation.params import PROPOSED_EXPLORATION
from stock_risk.domain.evaluation.series import EndReason
from stock_risk.domain.numeric import InvalidInputError
from tests.unit.evaluation.helpers import load_fixtures, make_series, run_case


def test_B26_common_mask_start_and_count() -> None:
    case = load_fixtures()["B"]["B26"]
    series = make_series(case)
    expected = case["expected_mask"]
    days = series.calendar.days
    assert series.segments[1].start == days[expected["second_segment_start"]]
    starts = [
        next(d for d in seg.bar_days if series.mask[series.index(d)]) for seg in series.segments
    ]
    assert starts == [
        days[expected["first_segment_mask_start"]],
        days[expected["second_segment_mask_start"]],
    ]
    assert sum(series.mask) == expected["mask_day_count"]
    assert series.break_days == (days[160],)
    assert series.segment_ids[160] is None
    assert series.segment_counts[161] == 1
    assert series.valid_ordinals[161] == 150
    assert not any(series.mask[150:161])


def test_atr20_requires_21_bars_and_restarts_after_break() -> None:
    series = make_series(load_fixtures()["B"]["B26"])
    assert all(x is None for x in series.atr20[:20])
    assert series.atr20[20] == Decimal(1)
    assert all(x is None for x in series.atr20[150:181])
    assert series.atr20[181] == Decimal(1)


def test_atr_uses_previous_valid_close_across_short_gap() -> None:
    case = load_fixtures()["A"]["A01"]
    rows = case["bars"][:101] + [None, ["95", "95.5", "89.5", "90"]]
    series = make_series({**case, "bars": rows})
    # 最近 20 个 TR = 18×1 + 6 + 6；缺价日不占位，前收为 95。
    assert series.atr20[102] == Decimal("1.5")
    assert series.segment_counts[101] == 101
    assert series.segment_counts[102] == 102
    assert series.valid_days[101] == series.calendar.days[102]
    assert series.valid_ordinals[101] is None
    assert not series.mask[101]


@pytest.mark.parametrize("reason", list(EndReason))
def test_right_censor_reason_and_prefix_completeness(reason: EndReason) -> None:
    case = {**load_fixtures()["A"]["A30"], "end_reason": reason.value}
    _, chain = run_case(case)
    event = chain.events[0]
    assert event.censor == frozenset({f"RIGHT_{reason.value}"})
    assert event.is_prefix_complete
    assert event.dependency_pending is not None
    assert event.dependency_pending.valid_days == 11
    assert chain.counts["dependency_pending_days"] == 11


def test_series_and_events_ignore_callers_decimal_context() -> None:
    case = load_fixtures()["A"]["A19d1"]
    original, chain = run_case(case)
    with localcontext() as context:
        context.prec = 5
        context.rounding = "ROUND_UP"
        # 日线已由入库量化函数生成；这里只检查评价层，不改变既有 numeric.py 的上下文契约。
        actual_series = replace(original)
        params = EventParameters(PROPOSED_EXPLORATION.rho_end, PROPOSED_EXPLORATION.w_q)
        actual = build_events(actual_series, chain.spec, params)
        assert context.prec == 5
        assert context.rounding == "ROUND_UP"
    assert actual_series == original
    assert actual == chain


def test_series_rejects_misaligned_bars() -> None:
    series = make_series(load_fixtures()["A"]["A01"])
    with pytest.raises(InvalidInputError, match="逐日对齐"):
        replace(series, bars=series.bars[:-1])
    with pytest.raises(InvalidInputError, match="日期与日历不一致"):
        replace(series, bars=tuple(reversed(series.bars)))
