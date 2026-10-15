"""回放结果 → episodes 行（每次预警一行，T04b；第 1A.7 节的关联字段）。纯函数。

从 replay_outputs 拆出（T08a D-03）。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

from stock_risk.application.replay import SymbolReplay
from stock_risk.application.replay_outputs import EPISODE_COLUMNS, cell, closes_by_day
from stock_risk.domain.lifecycle.model import EndReason, Event, EventType
from stock_risk.domain.lifecycle.restart import EpisodeSpan, days_since_prev_release


@dataclass(frozen=True, slots=True)
class EpisodeRow:
    episode_id: str
    start_date: date
    start_close: float | None
    start_reasons: tuple[str, ...]
    initial_level: str
    low_date: date | None
    low_close: float | None
    first_stabilized: tuple[date, float | None] | None
    first_obs: tuple[date, float | None] | None
    downgrade: tuple[date, float | None] | None
    end_date: date | None
    end_close: float | None
    end_reason: EndReason | None  # None：回放结束时仍在进行（不应出现，截尾会给出原因）
    n_recovery_obs: int
    n_revoked: int
    n_recovery_fail: int
    n_escalate: int
    duration_trading_days: int  # 自启动日到终止日（含两端）的交易日数
    observed_valid_days: int  # 其中有日线的交易日数
    entry_path: str | None = None  # v0.8：S / R / L
    delta_start: float | None = None  # v0.8：P_ep − 启动收盘
    prev_episode_id: str | None = None  # 第 1A.7 节：上一次预警
    days_since_prev_release: int | None = None  # 距上一次解除的有效交易日数（上一次不是解除时为空）
    start_below_prev_l_ep: bool | None = None  # 本次启动收盘是否低于上一次的 L_ep


def episode_rows(rep: SymbolReplay) -> list[EpisodeRow]:
    closes = closes_by_day(rep)
    days = [r.record.day for r in rep.result.results]
    by_eid: dict[str, list[Event]] = {}
    for e in rep.result.events:
        if e.type is not EventType.DATA_ISSUE:
            by_eid.setdefault(e.identity[0], []).append(e)
    out = [_episode_row(rep, eid, evs, days, closes) for eid, evs in by_eid.items()]
    return _link_previous(out, days, closes)


def _end_of(
    rep: SymbolReplay, eid: str, evs: list[Event], days: list[date]
) -> tuple[date | None, EndReason | None]:
    """终止日与原因：RELEASE、DATA_BREAK，或截尾（观察终点），或尚未结束（None）。"""
    end_ev = next((e for e in evs if e.type in (EventType.RELEASE, EventType.DATA_BREAK)), None)
    if end_ev is not None:
        kind = EndReason.RELEASE if end_ev.type is EventType.RELEASE else EndReason.DATA_BREAK
        return end_ev.day, kind
    if rep.result.censored_episode == eid:
        return days[-1], rep.result.results[-1].record.end_reason
    return None, None


def _episode_row(
    rep: SymbolReplay,
    eid: str,
    evs: list[Event],
    days: list[date],
    closes: Mapping[date, float],
) -> EpisodeRow:
    results = rep.result.results
    start = next(e for e in evs if e.type is EventType.START)

    def first(kind: EventType) -> tuple[date, float | None] | None:
        e = next((x for x in evs if x.type is kind), None)
        return None if e is None else (e.day, closes.get(e.day))

    end_day, reason = _end_of(rep, eid, evs, days)
    recs = [r.record for r in results if r.record.episode_id == eid and r.record.l_ep]
    lo, hi = days.index(start.day), days.index(end_day) if end_day else len(days) - 1
    span = results[lo : hi + 1]
    count = {k: sum(1 for e in evs if e.type is k) for k in EventType}
    return EpisodeRow(
        episode_id=eid,
        start_date=start.day,
        start_close=closes.get(start.day),
        start_reasons=tuple(start.basis["reasons"]),
        initial_level=start.basis["level"],
        low_date=recs[-1].l_ep_day if recs else None,
        low_close=recs[-1].l_ep if recs else None,
        first_stabilized=first(EventType.STABILIZED),
        first_obs=first(EventType.RECOVERY_OBS),
        downgrade=first(EventType.DOWNGRADE),
        end_date=end_day,
        end_close=closes.get(end_day) if end_day else None,
        end_reason=reason,
        n_recovery_obs=count[EventType.RECOVERY_OBS],
        n_revoked=count[EventType.RECOVERY_REVOKED],
        n_recovery_fail=count[EventType.RECOVERY_FAIL],
        n_escalate=count[EventType.ESCALATE],
        duration_trading_days=len(span),
        observed_valid_days=sum(1 for r in span if r.record.day in closes),
        entry_path=(start.basis.get("v08") or {}).get("entry_path"),
        delta_start=start.basis["p_ep"] - start.basis["close"],
    )


def _link_previous(
    rows: list[EpisodeRow], days: list[date], closes: Mapping[date, float]
) -> list[EpisodeRow]:
    """第 1A.7 节的关联字段：只是输出字段，不改变状态机。距上一次解除的口径见
    domain.lifecycle.restart（T08a C-06）。"""
    valid = [d for d in days if d in closes]
    linked: list[EpisodeRow] = []
    prev: EpisodeRow | None = None
    for row in sorted(rows, key=lambda r: r.start_date):
        if prev is None:
            linked.append(row)
        else:
            span = EpisodeSpan(prev.start_date, prev.end_date, prev.end_reason)
            gap = days_since_prev_release(span, row.start_date, valid)
            below = (
                None
                if row.start_close is None or prev.low_close is None
                else row.start_close < prev.low_close
            )
            linked.append(
                replace(
                    row,
                    prev_episode_id=prev.episode_id,
                    days_since_prev_release=gap,
                    start_below_prev_l_ep=below,
                )
            )
        prev = row
    return linked


def episode_csv_rows(rep: SymbolReplay, run_id: str, rows: Iterable[EpisodeRow]) -> list[dict]:
    def pair(p: tuple[date, float | None] | None) -> tuple[Any, Any]:
        return ("", "") if p is None else (p[0].isoformat(), cell(p[1]))

    out = []
    for e in rows:
        st, ob, dg = pair(e.first_stabilized), pair(e.first_obs), pair(e.downgrade)
        out.append(
            dict(
                zip(
                    EPISODE_COLUMNS,
                    [
                        run_id,
                        rep.candidate_id,
                        rep.target.label,
                        e.episode_id,
                        e.start_date.isoformat(),
                        cell(e.start_close),
                        json.dumps(list(e.start_reasons)),
                        e.initial_level,
                        cell(e.low_date),
                        cell(e.low_close),
                        *st,
                        *ob,
                        *dg,
                        cell(e.end_date),
                        cell(e.end_close),
                        cell(e.end_reason),
                        e.n_recovery_obs,
                        e.n_revoked,
                        e.n_recovery_fail,
                        e.n_escalate,
                        e.duration_trading_days,
                        e.observed_valid_days,
                        cell(e.entry_path),
                        cell(e.delta_start),
                        cell(e.prev_episode_id),
                        cell(e.days_since_prev_release),
                        cell(e.start_below_prev_l_ep),
                    ],
                    strict=True,
                )
            )
        )
    return out
