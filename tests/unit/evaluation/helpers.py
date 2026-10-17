"""T09 §6：仅负责夹具日序号、日期和日线转换，不推导事件期望值。"""

import json
from dataclasses import asdict
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.evaluation.event_model import (
    EventChain,
    EventKind,
    EventParameters,
    EventSetSpec,
)
from stock_risk.domain.evaluation.events import build_events
from stock_risk.domain.evaluation.method_series import AlertLevel, MethodEvent, MethodSeries
from stock_risk.domain.evaluation.params import PROPOSED_EXPLORATION
from stock_risk.domain.evaluation.series import EndReason, EvalSeries
from stock_risk.domain.market_data import DailyBar
from stock_risk.domain.numeric import quantize_price


def load_fixtures() -> dict[str, Any]:
    path = Path(__file__).parent / "fixtures" / "eval_fixtures_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def workdays(length: int) -> tuple[date, ...]:
    days: list[date] = []
    day = date(2011, 1, 3)
    while len(days) < length:
        if day.weekday() < 5:
            days.append(day)
        if len(days) < length:
            day += timedelta(days=1)
    return tuple(days)


def make_series(case: dict[str, Any], stop: int | None = None) -> EvalSeries:
    rows = case["bars"] if stop is None else case["bars"][: stop + 1]
    days = workdays(len(rows))
    bars: list[DailyBar | None] = []
    for day, row in zip(days, rows, strict=True):
        if row is None:
            bars.append(None)
        else:
            op, hi, lo, cl = map(quantize_price, row)
            bars.append(DailyBar(day, op, hi, lo, cl, 1, op, hi, lo, cl))
    return EvalSeries(
        TradingCalendar(days),
        tuple(bars),
        case.get("G", 10),  # 夹具约定 G = 10（conventions.G）；B 组夹具不单列
        PROPOSED_EXPLORATION.mask_min,
        EndReason(case.get("end_reason", "DATA_END")),
    )


def run_case(case: dict[str, Any], stop: int | None = None) -> tuple[EvalSeries, EventChain]:
    series = make_series(case, stop)
    spec = EventSetSpec(
        EventKind(case["event_set"]["kind"].upper()),
        Decimal(case["event_set"]["T"]),
        case["id"],
    )
    params = EventParameters(PROPOSED_EXPLORATION.rho_end, PROPOSED_EXPLORATION.w_q)
    return series, build_events(series, spec, params)


def assert_case(case_id: str) -> None:
    case = load_fixtures()["A"][case_id]
    series, chain = run_case(case)
    assert len(chain.events) == len(case["expected_events"])
    dates = {"s", "p", "end", "confirm", "report_trough", "k_star", "label_determinable"}
    # prefix_complete 由产品经理在夹具中显式给出（T09-E1R R4），不在测试中重算
    rename = {"C_p": "c_p", "end": "e", "main": "is_main", "prefix_complete": "is_prefix_complete"}
    for event, expected in zip(chain.events, case["expected_events"], strict=True):
        actual = asdict(event)
        for key, value in expected.items():
            if key in dates:
                value = series.calendar.days[value] if value is not None else None
            elif key == "C_p":
                value = Decimal(value)
            elif key in {"flags", "censor", "unevaluable"}:
                value = frozenset(value)
            elif key == "dependency_pending" and value is not None:
                value = {
                    "start_exclusive": series.calendar.days[value[0]],
                    "end_inclusive": series.calendar.days[value[1]],
                    "valid_days": value[2],
                }
            assert actual[rename.get(key, key)] == value, (case_id, key, actual, expected)
        assert "prefix_complete" in expected, (case_id, "夹具缺少 prefix_complete")
        assert event.a_s == (
            series.atr20[series.index(event.s)] if case["event_set"]["kind"] == "atr" else None
        )


def make_method(series: EvalSeries, name: str, spec: dict[str, Any]) -> MethodSeries:
    """夹具区段 → MethodSeries。

    区段格式为 [起日, 止日（含）, episode_id, 起始等级, {日: 新等级}]；区段外不在警。"""
    days = series.calendar.days
    levels: list[AlertLevel | None] = [None] * len(days)
    episodes: list[str | None] = [None] * len(days)
    for start, end, episode, level, changes in spec["spans"]:
        current = AlertLevel(level)
        for i in range(start, end + 1):
            current = AlertLevel(changes[str(i)]) if str(i) in changes else current
            levels[i], episodes[i] = current, episode
    events = tuple(
        MethodEvent(kind, days[day], episode) for kind, day, episode, *_ in spec["events"]
    )
    return MethodSeries(name, days, tuple(levels), tuple(episodes), events)


def day_or_none(series: EvalSeries, value: int | None) -> date | None:
    return series.calendar.days[value] if value is not None else None
