"""夹具测试的共用构造。期望值一律来自 10、12、13 号文档，不由引擎输出回填。

- S 组：按表给定的指标实现（TableIndicators）。每个夹具前置一个峰值日 p（收盘 100、ATR 2），
  历史长度视为充足（prior_valid_bars 足够大）。开盘取前一日收盘，使 R2 不触发
  （12 号第 7.1 节共同前提）。
- F 组：13 号第 2 节的完整日线构造，由日线计算指标端到端运行。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.indicator_source import TableIndicators, TableRow
from stock_risk.domain.lifecycle.invariants import check_invariants
from stock_risk.domain.lifecycle.model import DailyRecord, Event, EventType, StepResult
from stock_risk.domain.lifecycle.params import (
    AtrBasis,
    Candidate,
    Confirm,
    EngineParams,
    ExplorationAssumptions,
    ReleasePath,
)
from stock_risk.domain.lifecycle.runner import RunResult, run_bars, run_with_source
from stock_risk.domain.market_data import DailyBar
from stock_risk.domain.observation import DayPresence

SYMBOL = "TEST"
START_DAY = date(2009, 1, 5)  # 13 号第 2 节：自 2009-01-05（周一）起的连续工作日

# 13 号第 1 节"共同前提"（夹具自己的参数）：
# k₁ = 2、k₂ = 4、K = 3、d = 3、u_rel = 3、K_stab = 8、G = 3、φ = 0.5
FIXTURE_ASSUMPTIONS = ExplorationAssumptions(phi=0.5, k_stab=8, gap_break=3)


def params(
    confirm: Confirm = Confirm.SMA,
    release: ReleasePath = ReleasePath.TWO_STAGE,
    basis: AtrBasis = AtrBasis.PEAK,
    **over: Any,
) -> EngineParams:
    base = dict(n=20, k1=2, k2=4, k=3, d=3, u_rel=3)
    base.update(over)
    cand = Candidate(**base, confirm=confirm, release_path=release, atr_basis=basis)
    return EngineParams(candidate=cand, assumptions=FIXTURE_ASSUMPTIONS)


def weekdays(start: date, count: int) -> list[date]:
    out, d = [], start
    while len(out) < count:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


@dataclass(frozen=True)
class Run:
    """一次夹具运行：按标签取当日结果。"""

    labels: dict[str, date]
    result: RunResult
    params: EngineParams

    def rec(self, label: str) -> DailyRecord:
        return self.step(label).record

    def step(self, label: str) -> StepResult:
        day = self.labels[label]
        return next(r for r in self.result.results if r.record.day == day)

    def events(self, label: str) -> list[Event]:
        return list(self.step(label).events)

    def types(self, label: str) -> list[EventType]:
        return sorted(e.type for e in self.events(label))

    def event(self, label: str, kind: EventType) -> Event:
        (e,) = [e for e in self.events(label) if e.type is kind]
        return e

    def invariants(self) -> list[str]:
        return check_invariants(self.result.results, self.params)


def s_run(
    rows: list[tuple[str, float | None, float | None]],
    p: EngineParams,
    atr: dict[str, float] | None = None,
    opens: dict[str, float] | None = None,
    peak_atr: float = 2.0,
) -> Run:
    """S 组：rows = [(标签, 收盘或 None（缺价）, SMA10 或 None)]，前置峰值日 p。"""
    labels = ["p", *[r[0] for r in rows]]
    days = weekdays(START_DAY, len(labels))
    lab = dict(zip(labels, days, strict=True))
    atr = atr or {}
    opens = opens or {}
    table: dict[date, TableRow] = {days[0]: TableRow(100.0, 100.0, atr=peak_atr)}
    prev_close = 100.0
    for (label, close, sma), day in zip(rows, days[1:], strict=True):
        if close is None:
            continue
        o = opens.get(label, prev_close)
        table[day] = TableRow(o, close, atr=atr.get(label, 2.0), sma=sma)
        prev_close = close
    cal = TradingCalendar.from_days(days)
    src = TableIndicators(cal, table, n=p.candidate.n, prior_valid_bars=10_000)
    return Run(lab, run_with_source(SYMBOL, days, src, p), p)


def f_bars(after_peak: list[float]) -> list[DailyBar]:
    """13 号第 2 节：90 根收盘 100（峰值日是最后一根）；O_t = C_{t−1}（第一根 O = C）；
    H = max(O, C) + 1，L = min(O, C) − 1；复权价同原始价；成交量 1,000,000。"""
    closes = [100.0] * 90 + after_peak
    days = weekdays(START_DAY, len(closes))
    bars, prev = [], None
    for day, c in zip(days, closes, strict=True):
        o = c if prev is None else prev
        h, lo = max(o, c) + 1, min(o, c) - 1
        bars.append(DailyBar(day, o, h, lo, c, 1_000_000, o, h, lo, c))
        prev = c
    return bars


def f_run(prefix: str, after_peak: list[float], p: EngineParams, upto: int | None = None) -> Run:
    """F 组：标签 prefix0 为峰值日（F1 另用 c−1 记峰值日，见调用处）。upto：只回放到第 upto 根。"""
    bars = f_bars(after_peak)
    if upto is not None:
        bars = bars[: upto + 1]
    cal = TradingCalendar.from_days(b.day for b in bars)
    presence = [DayPresence(b.day, True) for b in bars]
    result = run_bars(SYMBOL, cal, presence, bars, p, replay_start=bars[0].day)
    labels = {f"{prefix}{i}": bars[89 + i].day for i in range(len(bars) - 89)}
    return Run(labels, result, p)


def kinds(run: Run) -> dict[date, list[EventType]]:
    out: dict[date, list[EventType]] = {}
    for r in run.result.results:
        if r.events:
            out[r.record.day] = sorted(e.type for e in r.events)
    return out
