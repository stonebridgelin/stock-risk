"""生命周期引擎：纯函数步进（10 号规格 v0.7.1 第 1.2 节 A / B / C）。

step(state, day, src, params, replay_start) -> StepResult：
- 输入：昨日状态、当日（由 src.has_bar 判定有日线 / 缺价）、指标接口（I5，T01 两种实现均可）。
- src 必须只覆盖当前历史段（由 history_segments.split_segments 切分），引擎用 src.valid_bar_count
  判定 W + N。
- 不读文件、不取当前时间、不依赖全局状态；状态对象不可变。
- 回放起点之前只运行 A（缺价计数、历史段），不运行 B、C，不产生任何事件（I2）。
- 不实现：第 12 步观察标签（I8）；第 1.6 节状态修正与衔接；投递、冷却、文案。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Any

from stock_risk.domain.indicator_source import IndicatorSource, compute_r2
from stock_risk.domain.lifecycle.episode_rules import observation_reset
from stock_risk.domain.lifecycle.episode_step import step_episode
from stock_risk.domain.lifecycle.model import (
    DataStatus,
    Display,
    EndReason,
    Episode,
    Event,
    EventType,
    Level,
    ResetReason,
    SecurityState,
    StepResult,
)
from stock_risk.domain.lifecycle.params import AtrBasis, EngineParams
from stock_risk.domain.lifecycle.records import build_record
from stock_risk.domain.numeric import InvalidInputError, Kind, ge


def episode_id_for(symbol: str, start_day: date) -> str:
    """I19：episode_id 由（代码，START 信号日）确定，重跑不变。"""
    return f"{symbol}:{start_day.isoformat()}"


def initial_state(symbol: str) -> SecurityState:
    return SecurityState(symbol=symbol)


def _display(ep: Episode | None) -> Display:
    if ep is None:
        return Display.NORMAL
    return Display.ALERT if ep.level is Level.ALERT else Display.WATCH


def step(
    state: SecurityState,
    day: date,
    src: IndicatorSource,
    params: EngineParams,
    replay_start: date | None = None,
) -> StepResult:
    in_replay = replay_start is None or day >= replay_start
    if not src.has_bar(day):
        return _missing_day(state, day, params, in_replay)

    # A 第 2 步：有日线
    g = params.assumptions.gap_break
    from_break = state.m >= g + 1
    seg_start = day if (state.seg_start is None or from_break) else state.seg_start
    state = replace(state, m=0, seg_start=seg_start)
    # A 第 3 步：历史段不足 W + N 条
    if src.valid_bar_count(day) < params.fixed.w + params.candidate.n:
        if state.episode is not None:
            raise InvalidInputError(f"{day}: 有进行中的预警但历史段不足 W + N（历史段切分有误）")
        state = replace(state, display=Display.INSUFFICIENT)
        rec = build_record(
            state,
            day,
            params,
            in_replay=in_replay,
            data_status=DataStatus.INSUFFICIENT,
            ep=None,
            extra={"transition": "T4" if from_break else "T5"},
        )
        return StepResult(state, (), rec)
    if not in_replay:
        state = replace(state, display=Display.NORMAL)
        rec = build_record(
            state,
            day,
            params,
            in_replay=False,
            data_status=DataStatus.HAS_BAR,
            ep=None,
            extra={"transition": "T4" if from_break else None},
        )
        return StepResult(state, (), rec)
    if state.episode is None:
        return _no_episode(state, day, src, params, from_break)
    return _with_episode(state, day, src, params)


def _missing_day(
    state: SecurityState, day: date, params: EngineParams, in_replay: bool
) -> StepResult:
    """A 第 1 步（T1–T3；E-02）。"""
    g = params.assumptions.gap_break
    m = state.m + 1
    ep = state.episode
    events: list[Event] = []
    extra: dict[str, Any] = {}
    issue = Event(
        EventType.DATA_ISSUE, day, (state.symbol, EventType.DATA_ISSUE.value, day), None, {"m": m}
    )
    if m <= g:
        if ep is not None:
            if ep.prev_confirm:
                events.append(
                    Event(
                        EventType.RECOVERY_REVOKED,
                        day,
                        (ep.episode_id, ep.cycle_no, ep.obs_n),
                        "T1",
                        {"reason": "DATA_GAP"},
                    )
                )
            ep = replace(observation_reset(ep, day), prev_confirm=False)
            extra["reset_reasons"] = (ResetReason.DATA_GAP,)
        if in_replay:
            events.append(replace(issue, transition="T1"))
        state = replace(state, m=m, episode=ep)
        extra["transition"] = "T1"
        status = DataStatus.MISSING
    elif m == g + 1:
        if ep is not None:
            events.append(
                Event(
                    EventType.DATA_BREAK,
                    day,
                    (ep.episode_id, EventType.DATA_BREAK.value, day),
                    "T2",
                    {"m": m},
                )
            )
            extra["end_reason"] = EndReason.DATA_BREAK
            extra["episode_id"] = ep.episode_id
        if in_replay:
            events.append(replace(issue, transition="T2"))
        state = replace(state, m=m, episode=None, last_release=None, display=Display.DATA_BREAK)
        extra["transition"] = "T2"
        status = DataStatus.DATA_BREAK
    else:
        state = replace(state, m=m, display=Display.DATA_BREAK)
        extra["transition"] = "T3"
        status = DataStatus.DATA_BREAK
    rec = build_record(
        state, day, params, in_replay=in_replay, data_status=status, ep=state.episode, extra=extra
    )
    return StepResult(state, tuple(events), rec)


def _no_episode(
    state: SecurityState, day: date, src: IndicatorSource, params: EngineParams, from_break: bool
) -> StepResult:
    """B 第 4–5 步（T6–T8）。"""
    cand, tol = params.candidate, params.tol
    c = src.close(day)
    p_max, p_day = src.window_max_close(day, params.fixed.w, state.last_release)
    atr_peak = src.atr(p_day, cand.n)
    if atr_peak is None:
        raise InvalidInputError(f"{day}: 峰值日 {p_day} 的 ATR_{cand.n} 不可计算")
    a_start = atr_peak
    if cand.atr_basis is AtrBasis.MIN:
        atr_t = src.atr(day, cand.n)
        if atr_t is None:
            raise InvalidInputError(f"{day}: ATR_{cand.n} 不可计算")
        a_start = min(atr_peak, atr_t)
    d_t = (p_max - c) / a_start
    r2 = compute_r2(src, day, cand.n, params.fixed.g, tol)
    extra: dict[str, Any] = {
        "d_t": d_t,
        "a_start": a_start,
        "r2_status": r2.status.value,
        "r2_score": r2.gap_score,
        "r2_triggered": r2.triggered,
    }
    reasons: list[str] = []
    if ge(d_t, cand.k2, Kind.RATIO, tol):
        level, transition, reasons = Level.ALERT, "T6", ["R1_GE_K2"]
    elif ge(d_t, cand.k1, Kind.RATIO, tol) or r2.triggered:
        level, transition = Level.WATCH, "T7"
        if ge(d_t, cand.k1, Kind.RATIO, tol):
            reasons.append("R1_GE_K1")
    else:
        state = replace(state, display=Display.NORMAL)
        extra["transition"] = "T4" if from_break else "T8"
        rec = build_record(
            state, day, params, in_replay=True, data_status=DataStatus.HAS_BAR, ep=None, extra=extra
        )
        return StepResult(state, (), rec)
    if r2.triggered:
        reasons.append("GAP")
    l_ep, l_day = src.min_close(p_day, day)
    eid = episode_id_for(state.symbol, day)
    ep = Episode(eid, day, level, p_max, p_day, a_start, l_ep, l_day, obs_start=day, cycle_no=1)
    start = Event(
        EventType.START,
        day,
        (eid, EventType.START.value, day),
        transition,
        {
            "reasons": reasons,
            "level": level.name,
            "d_t": d_t,
            "a_start": a_start,
            "p_ep": p_max,
            "p_ep_day": p_day,
            "l_ep": l_ep,
            "close": c,
            "r2_status": r2.status.value,
            "r2_score": r2.gap_score,
        },
    )
    state = replace(state, episode=ep, display=_display(ep))
    extra.update(transition=transition, d_ep=d_t)
    rec = build_record(
        state, day, params, in_replay=True, data_status=DataStatus.HAS_BAR, ep=ep, extra=extra
    )
    return StepResult(state, (start,), rec)


def _with_episode(
    state: SecurityState, day: date, src: IndicatorSource, params: EngineParams
) -> StepResult:
    """C 第 6–11 步（T9–T16）。"""
    assert state.episode is not None
    out = step_episode(state.episode, day, src, params)
    extra: dict[str, Any] = {**out.fields, "transition": out.transition}
    if out.released:
        # 解除日记录被解除预警的冻结字段；等级为空，计数与 H_d 清空
        state = replace(state, episode=None, last_release=day, display=Display.NORMAL)
        extra.update(end_reason=EndReason.RELEASE, level=None, dc=None, rc=None, xc=None, h_d=None)
    else:
        state = replace(state, episode=out.episode, display=_display(out.episode))
    rec = build_record(
        state,
        day,
        params,
        in_replay=True,
        data_status=DataStatus.HAS_BAR,
        ep=out.episode,
        extra=extra,
    )
    return StepResult(state, out.events, rec)


def finalize(state: SecurityState, reason: EndReason) -> tuple[SecurityState, str | None]:
    """观察终点（I3）：进行中的预警标为截尾（LISTING_END 或 EXPLORATION_END）。

    截尾不是事件，不产生 RELEASE 或 DATA_BREAK。返回（预警清空后的状态，被截尾的 episode_id）。
    """
    if reason not in (EndReason.LISTING_END, EndReason.EXPLORATION_END):
        raise ValueError(f"截尾原因只能是 LISTING_END 或 EXPLORATION_END: {reason}")
    if state.episode is None:
        return state, None
    return replace(state, episode=None), state.episode.episode_id
