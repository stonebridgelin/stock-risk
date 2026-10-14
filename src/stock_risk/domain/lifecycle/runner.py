"""逐日运行引擎（纯函数，供夹具与以后的回放共用）。

- run_with_source：用给定的指标接口（例如按表给定的 S 组夹具）逐日运行；
  调用方保证 src 只覆盖一个历史段。
- run_bars：由日线逐日运行；按 G 切分历史段，每段建一个 ComputedIndicators，指标在新段内重新积累。
到达观察终点时可给出截尾原因（I3）：进行中的预警记为截尾，写在最后一天的诊断记录里。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date

from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.indicator_source import ComputedIndicators, IndicatorSource
from stock_risk.domain.lifecycle.engine import finalize, initial_state, step
from stock_risk.domain.lifecycle.history_segments import split_segments
from stock_risk.domain.lifecycle.model import EndReason, Event, SecurityState, StepResult
from stock_risk.domain.lifecycle.params import EngineParams
from stock_risk.domain.market_data import DailyBar
from stock_risk.domain.observation import DayPresence


@dataclass(frozen=True, slots=True)
class RunResult:
    results: tuple[StepResult, ...]
    final_state: SecurityState
    censored_episode: str | None

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(e for r in self.results for e in r.events)


def _censor(
    results: list[StepResult], state: SecurityState, end_reason: EndReason | None
) -> RunResult:
    censored = None
    if end_reason is not None and results:
        state, censored = finalize(state, end_reason)
        if censored is not None:
            last = results[-1]
            results[-1] = replace(last, record=replace(last.record, end_reason=end_reason))
    return RunResult(tuple(results), state, censored)


def run_with_source(
    symbol: str,
    days: Sequence[date],
    src: IndicatorSource,
    params: EngineParams,
    replay_start: date | None = None,
    state: SecurityState | None = None,
    end_reason: EndReason | None = None,
) -> RunResult:
    state = state or initial_state(symbol)
    results: list[StepResult] = []
    for d in days:
        r = step(state, d, src, params, replay_start)
        results.append(r)
        state = r.state
    return _censor(results, state, end_reason)


def run_bars(
    symbol: str,
    calendar: TradingCalendar,
    presence: Sequence[DayPresence],
    bars: Sequence[DailyBar],
    params: EngineParams,
    replay_start: date | None = None,
    end_reason: EndReason | None = None,
) -> RunResult:
    """presence：观察区间内逐日"有日线 / 缺价"；
    bars：观察区间内的有效日线（复权价，已量化与更正）。"""
    by_day = {b.day: b for b in bars}
    segments = split_segments(presence, params.assumptions.gap_break)
    sources = [ComputedIndicators(calendar, [by_day[d] for d in s.bar_days]) for s in segments]
    state = initial_state(symbol)
    results: list[StepResult] = []
    seg_i = 0
    for p in presence:
        if p.has_bar:
            while seg_i + 1 < len(segments) and p.day >= segments[seg_i + 1].start:
                seg_i += 1
        src = sources[seg_i] if sources else ComputedIndicators(calendar, [])
        r = step(state, p.day, src, params, replay_start)
        results.append(r)
        state = r.state
    return _censor(results, state, end_reason)
