"""图表数据（T04b chart.html）。应用层只组织数据，由基础设施层用 plotly 绘制。

主图：复权收盘、状态色带、事件标记（悬停显示事件与依据）、L_ep、确认线、强恢复门槛线、更正区间与缺价日。
副图：D_T / D_ep，以及 SPY、QQQ 的 63 日回撤（只作参照，未应用背景标签，I8）。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from stock_risk.application.replay import SymbolReplay
from stock_risk.application.replay_outputs import closes_by_day
from stock_risk.domain.lifecycle.model import Display
from stock_risk.domain.lifecycle.params import FIXED
from stock_risk.domain.market_data import DailyBar

BACKGROUND_NOTE = "SPY、QQQ 的 63 日回撤：参照显示，不是大盘状态，不进入信号"
BAND_STATES = (Display.WATCH, Display.ALERT, Display.DATA_BREAK, Display.INSUFFICIENT)


@dataclass(frozen=True, slots=True)
class ChartEvent:
    day: date
    type: str
    y: float | None
    text: str


@dataclass(frozen=True, slots=True)
class ChartData:
    title: str
    days: tuple[date, ...]
    close: tuple[float | None, ...]
    bands: tuple[tuple[date, date, str], ...]  # (起, 止, 显示状态)
    events: tuple[ChartEvent, ...]
    l_ep: tuple[float | None, ...]
    confirm_line: tuple[float | None, ...]
    strong_threshold: tuple[float | None, ...]
    d_t: tuple[float | None, ...]
    d_ep: tuple[float | None, ...]
    background: dict[str, tuple[float | None, ...]]  # 代码 → 63 日回撤（与 days 对齐）
    corrections: tuple[tuple[date, date, str], ...]
    missing_days: tuple[date, ...]
    note: str = BACKGROUND_NOTE


def drawdown_63(bars: Sequence[DailyBar], window: int = FIXED.w) -> dict[date, float]:
    """回撤 = C_t / 最近 window 条有效日线（含当日）的最高收盘 − 1（复权价）。"""
    out: dict[date, float] = {}
    closes = [b.adj_close for b in bars]
    for i, b in enumerate(bars):
        peak = max(closes[max(0, i - window + 1) : i + 1])
        out[b.day] = b.adj_close / peak - 1
    return out


def _bands(days: Sequence[date], displays: Sequence[Display]) -> tuple[tuple[date, date, str], ...]:
    out: list[tuple[date, date, str]] = []
    start = 0
    for i in range(1, len(days) + 1):
        if i == len(days) or displays[i] != displays[start]:
            if displays[start] in BAND_STATES:
                out.append((days[start], days[i - 1], displays[start].value))
            start = i
    return tuple(out)


def build_chart_data(
    rep: SymbolReplay, run_id: str, background: dict[str, dict[date, float]]
) -> ChartData:
    results = rep.result.results
    recs = [r.record for r in results]
    days = tuple(r.day for r in recs)
    closes = closes_by_day(rep)
    close = tuple(closes.get(d) for d in days)
    events = tuple(
        ChartEvent(
            e.day,
            e.type.value,
            closes.get(e.day),
            f"{e.type.value} {e.transition or ''}<br>"
            + json.dumps({k: str(v) for k, v in e.basis.items()}, ensure_ascii=False)[:400],
        )
        for e in rep.result.events
    )
    c = rep.params.candidate
    title = (
        f"{rep.target.label} · 候选 {rep.candidate_id}（{c.confirm} / {c.release_path} / "
        f"{c.atr_basis}，N={c.n}、k₁={c.k1}、k₂={c.k2}、K={c.k}、d={c.d}、u_rel={c.u_rel}）"
        f" · run {run_id}"
    )
    corrections = tuple(
        (x.start, x.end, f"更正 {x.method}（{x.ruling}）") for x in rep.prepared.data.corrections
    )
    return ChartData(
        title=title,
        days=days,
        close=close,
        bands=_bands(days, [r.display for r in recs]),
        events=events,
        l_ep=tuple(r.l_ep if r.level is not None else None for r in recs),
        confirm_line=tuple(r.confirm_line for r in recs),
        strong_threshold=tuple(r.strong_threshold for r in recs),
        d_t=tuple(r.d_t for r in recs),
        d_ep=tuple(r.d_ep if r.level is not None else None for r in recs),
        background={k: tuple(v.get(d) for d in days) for k, v in background.items()},
        corrections=corrections,
        missing_days=tuple(r.day for r in recs if r.data_status.value == "MISSING"),
    )
