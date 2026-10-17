"""T09-E1 属性（T09-E1R 修订）：价格缩放；按标签可确定日的前缀属性（R1）与按确认日过滤的反例。"""

from dataclasses import replace
from decimal import Decimal

import pytest

from tests.unit.evaluation.helpers import load_fixtures, run_case

PCT_CASES = [
    name for name, case in load_fixtures()["A"].items() if case["event_set"]["kind"] == "pct"
]


@pytest.mark.parametrize("case_id", PCT_CASES)
def test_pct_scale_by_ten(case_id: str) -> None:
    case = load_fixtures()["A"][case_id]
    _, original = run_case(case)
    scaled = {
        **case,
        "bars": [
            [str(Decimal(value) * 10) for value in row] if row is not None else None
            for row in case["bars"]
        ],
    }
    _, result = run_case(scaled)
    assert result.events == tuple(replace(e, c_p=e.c_p * 10) for e in original.events)


def test_confirm_day_filter_is_wrong_condition_A29b() -> None:
    """T09-E1R R1 的反例：按"确认日 ≤ t"过滤前缀会出错，此差异确实存在。

    A29b 截断到 t = 124：完整运行中有一个确认日 ≤ 124、但标签可确定日 = 140 的事件
    （它要等前序事件在第 140 日确认后、从 e 的次日回放才建立，D-10、D-14）；前缀运行中没有它。
    """
    case = load_fixtures()["A"]["A29b"]
    series, full = run_case(case)
    _, prefix = run_case(case, stop=124)
    day = series.calendar.days
    late = [
        e
        for e in full.events
        if e.confirm is not None and e.confirm <= day[124] and e.label_determinable == day[140]
    ]
    assert len(late) == 1
    assert late[0] not in prefix.events
    assert all((e.s, e.p) != (late[0].s, late[0].p) for e in prefix.events)


# T09-E1R 第 4 节第 2 条：这几个夹具在每一个 t 上检验，其余在关键日前后 1 日检验
EVERY_DAY = {"A29b", "A07", "A_DB", "A_M3"}


def _cutoffs(case_id: str, series, full) -> list[int]:  # type: ignore[no-untyped-def]
    last = len(series.bars) - 1
    if case_id in EVERY_DAY:
        return list(range(last + 1))
    boundaries = set(series.break_days)
    for event in full.events:
        boundaries.update(d for d in (event.confirm, event.label_determinable) if d is not None)
    ts = {series.index(d) + offset for d in boundaries for offset in (-1, 0, 1)}
    return sorted(t for t in ts if 0 <= t <= last)


@pytest.mark.parametrize("case_id", list(load_fixtures()["A"]))
def test_prefix_by_label_determinable_day(case_id: str) -> None:
    """T09-E1R R1：截断到第 t 日运行，标签可确定日 ≤ t 的事件与完整运行中标签可确定日 ≤ t 的
    事件全字段相同。"""
    case = load_fixtures()["A"][case_id]
    series, full = run_case(case)
    for t in _cutoffs(case_id, series, full):
        _, prefix = run_case(case, stop=t)
        cutoff = series.calendar.days[t]

        def known(events, cutoff=cutoff):  # type: ignore[no-untyped-def]
            return tuple(
                e
                for e in events
                if e.label_determinable is not None and e.label_determinable <= cutoff
            )

        assert known(prefix.events) == known(full.events), (case_id, t)
