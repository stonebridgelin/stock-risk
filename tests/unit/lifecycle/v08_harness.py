"""v0.8 夹具（13 号第 6 节）的共用构造。期望值一律来自 13 号文档，不由引擎输出回填。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from decimal import Decimal
from typing import Any

from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.indicator_source import TableIndicators, TableRow
from stock_risk.domain.lifecycle.engine import episode_id_for
from stock_risk.domain.lifecycle.model import Display, Episode, Level, SecurityState
from stock_risk.domain.lifecycle.params import (
    AtrBasis,
    Candidate,
    Confirm,
    EngineParams,
    LongPath,
    RecScale,
    ReleaseHyst,
    ReleasePath,
    V08Switches,
)
from stock_risk.domain.lifecycle.runner import run_bars, run_with_source
from stock_risk.domain.market_data import DailyBar
from stock_risk.domain.observation import DayPresence
from tests.unit.lifecycle.harness import FIXTURE_ASSUMPTIONS, START_DAY, SYMBOL, Run, weekdays

# 13 号第 6.0 节开关简记
V0 = V08Switches()
A = V08Switches(x1=0.08, x2=0.15, t11_full=True, long_path=LongPath.X1)
B = replace(A, release_hyst=ReleaseHyst.EFFECTIVE)
V2 = V08Switches(release_hyst=ReleaseHyst.EFFECTIVE)


def params(
    switches: V08Switches,
    confirm: Confirm = Confirm.SMA,
    release: ReleasePath = ReleasePath.TWO_STAGE,
    basis: AtrBasis = AtrBasis.PEAK,
    **over: Any,
) -> EngineParams:
    base = dict(n=20, k1=2, k2=4, k=3, d=3, u_rel=3)
    base.update(over)
    cand = Candidate(**base, confirm=confirm, release_path=release, atr_basis=basis)
    return EngineParams(candidate=cand, assumptions=FIXTURE_ASSUMPTIONS, switches=switches)


@dataclass
class Prior:
    """注入的前态（13 号第 6.0 节"注入前态的共同约定"）。"""

    level: Level
    l_ep: float
    entry_path: str
    delta_start: float
    p_ep: float = 100.0
    atr_ep: float = 2.0
    downgraded: bool = False
    h_d: float | None = None
    since_obs: int = 5
    stabilized: bool = False
    obs_n: int = 0
    prev_confirm: bool = False
    cycle_no: int = 1


@dataclass
class Day:
    label: str
    close: float | None  # None = 缺价
    sma: float | None = None
    atr: float | None = 2.0
    open: float | None = None  # 默认取前一日收盘


@dataclass
class Table:
    days: list[Day]
    history: list[Day] = field(default_factory=list)  # 只用于指标与窗口，不逐日运行


def run_table(
    table: Table,
    p: EngineParams,
    prior: Prior | None = None,
    last_release_label: str | None = None,
    prior_valid_bars: int = 10_000,
) -> Run:
    """按表运行：history 中的日子只作为指标输入（窗口、ATR），days 中的日子逐日运行。

    有注入前态时，预警的启动日、峰值日、低点日、obs_start 记为 history 首日之前一日（推导：
    前态的具体日期不影响夹具的断言）。
    """
    allrows = [*table.history, *table.days]
    cal_days = weekdays(START_DAY, len(allrows) + 1)
    lab = {d.label: day for d, day in zip(allrows, cal_days[1:], strict=True)}
    rows: dict[date, TableRow] = {}
    prev = None
    for d in allrows:
        if d.close is None:
            continue
        o = d.open if d.open is not None else (prev if prev is not None else d.close)
        rows[lab[d.label]] = TableRow(o, d.close, atr=d.atr, sma=d.sma)
        prev = d.close
    cal = TradingCalendar.from_days(cal_days)
    src = TableIndicators(cal, rows, n=p.candidate.n, prior_valid_bars=prior_valid_bars)
    pre = cal_days[0]
    state = SecurityState(
        SYMBOL,
        seg_start=pre,
        m=0,
        last_release=lab[last_release_label] if last_release_label else None,
        display=Display.NORMAL,
    )
    if prior is not None:
        ep = Episode(
            episode_id_for(SYMBOL, pre),
            pre,
            prior.level,
            prior.p_ep,
            pre,
            prior.atr_ep,
            prior.l_ep,
            pre,
            obs_start=pre,
            cycle_no=prior.cycle_no,
            since_obs=prior.since_obs,
            downgraded=prior.downgraded,
            h_d=prior.h_d,
            stabilized=prior.stabilized,
            obs_n=prior.obs_n,
            prev_confirm=prior.prev_confirm,
            entry_path=prior.entry_path,
            delta_start=prior.delta_start,
        )
        state = replace(
            state,
            episode=ep,
            display=Display.ALERT if prior.level is Level.ALERT else Display.WATCH,
        )
    run_days = [lab[d.label] for d in table.days]
    return Run(lab, run_with_source(SYMBOL, run_days, src, p, state=state), p)


def decimal_bars(after_peak: list[Decimal]) -> list[DailyBar]:
    """13 号第 2 节的完整日线，价格按十进制精确构造后再转为 float。"""
    closes = [Decimal(100)] * 90 + after_peak
    days = weekdays(START_DAY, len(closes))
    bars, prev = [], None
    for d, c in zip(days, closes, strict=True):
        o = c if prev is None else prev
        h, lo = max(o, c) + 1, min(o, c) - 1
        f = [float(x) for x in (o, h, lo, c)]
        bars.append(DailyBar(d, f[0], f[1], f[2], f[3], 1_000_000, f[0], f[1], f[2], f[3]))
        prev = c
    return bars


def f_run_decimal(prefix: str, after_peak: list[Decimal], p: EngineParams) -> Run:
    bars = decimal_bars(after_peak)
    cal = TradingCalendar.from_days(b.day for b in bars)
    presence = [DayPresence(b.day, True) for b in bars]
    result = run_bars(SYMBOL, cal, presence, bars, p, replay_start=bars[0].day)
    labels = {f"{prefix}{i}": bars[89 + i].day for i in range(len(bars) - 89)}
    return Run(labels, result, p)


__all__ = [
    "A",
    "B",
    "V0",
    "V2",
    "Day",
    "Prior",
    "RecScale",
    "Table",
    "params",
    "run_table",
    "decimal_bars",
    "f_run_decimal",
]
