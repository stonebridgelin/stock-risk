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

from stock_risk.domain.indicator_source import IndicatorSource
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
from stock_risk.domain.lifecycle.params import EngineParams
from stock_risk.domain.lifecycle.records import build_record
from stock_risk.domain.lifecycle.start_paths import StartDecision, evaluate_start
from stock_risk.domain.numeric import InvalidInputError


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
    if not in_replay and state.episode is not None:  # T08a B-07
        raise InvalidInputError(f"{day}: 回放起点 {replay_start} 之前有进行中的预警")
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
    issue = Event(
        EventType.DATA_ISSUE, day, (state.symbol, EventType.DATA_ISSUE.value, day), None, {"m": m}
    )
    if m <= g:
        state, events, extra = _gap_day(state, day, m, in_replay, issue)
        status = DataStatus.MISSING
    elif m == g + 1:
        state, events, extra = _break_day(state, day, m, in_replay, issue)
        status = DataStatus.DATA_BREAK
    else:
        state, events, extra = replace(state, m=m, display=Display.DATA_BREAK), [], {}
        extra["transition"] = "T3"
        status = DataStatus.DATA_BREAK
    rec = build_record(
        state, day, params, in_replay=in_replay, data_status=status, ep=state.episode, extra=extra
    )
    return StepResult(state, tuple(events), rec)


def _gap_day(
    state: SecurityState, day: date, m: int, in_replay: bool, issue: Event
) -> tuple[SecurityState, list[Event], dict[str, Any]]:
    """T1：连续缺价不超过 G。有预警时撤销尚未结束的确认（DATA_GAP）并观察重置。"""
    ep, events = state.episode, []
    extra: dict[str, Any] = {"transition": "T1"}
    if ep is not None:
        if ep.prev_confirm and in_replay:
            key = (ep.episode_id, ep.cycle_no, ep.obs_n)
            basis = {"reason": "DATA_GAP"}
            events.append(Event(EventType.RECOVERY_REVOKED, day, key, "T1", basis))
        ep = replace(observation_reset(ep, day), prev_confirm=False)
        extra["reset_reasons"] = (ResetReason.DATA_GAP,)
    if in_replay:
        events.append(replace(issue, transition="T1"))
    return replace(state, m=m, episode=ep), events, extra


def _break_day(
    state: SecurityState, day: date, m: int, in_replay: bool, issue: Event
) -> tuple[SecurityState, list[Event], dict[str, Any]]:
    """T2：连续缺价达到 G + 1，数据中断。进行中的预警以 DATA_BREAK 结束，last_release 清空。"""
    ep, events = state.episode, []
    extra: dict[str, Any] = {"transition": "T2"}
    if ep is not None and in_replay:
        key = (ep.episode_id, EventType.DATA_BREAK.value, day)
        events.append(Event(EventType.DATA_BREAK, day, key, "T2", {"m": m}))
        extra["end_reason"] = EndReason.DATA_BREAK
        extra["episode_id"] = ep.episode_id
    if in_replay:
        events.append(replace(issue, transition="T2"))
    state = replace(state, m=m, episode=None, last_release=None, display=Display.DATA_BREAK)
    return state, events, extra


def _no_episode(
    state: SecurityState, day: date, src: IndicatorSource, params: EngineParams, from_break: bool
) -> StepResult:
    """B 第 4–5 步（T6–T8；v0.8 第 1A.3 节）。"""
    dec = evaluate_start(src, day, state.last_release, params)
    extra: dict[str, Any] = dict(dec.record)
    if dec.level is None:
        state = replace(state, display=Display.NORMAL)
        extra["transition"] = "T4" if from_break else "T8"
        extra.pop("d_ep", None)
        extra.pop("dd_ep", None)
        rec = build_record(
            state, day, params, in_replay=True, data_status=DataStatus.HAS_BAR, ep=None, extra=extra
        )
        return StepResult(state, (), rec)
    ep, start = _start_episode(state.symbol, day, src, dec)
    state = replace(state, episode=ep, display=_display(ep))
    extra["transition"] = dec.transition
    rec = build_record(
        state, day, params, in_replay=True, data_status=DataStatus.HAS_BAR, ep=ep, extra=extra
    )
    return StepResult(state, (start,), rec)


def _start_episode(
    symbol: str, day: date, src: IndicatorSource, dec: StartDecision
) -> tuple[Episode, Event]:
    """第 5 步：建立新预警（周期 1，obs_start = 当日），并给出 START 事件。"""
    c = src.close(day)
    l_ep, l_day = src.min_close(dec.p_ep_day, day)
    eid = episode_id_for(symbol, day)
    ep = Episode(
        eid,
        day,
        dec.level,
        dec.p_ep,
        dec.p_ep_day,
        dec.atr_ep,
        l_ep,
        l_day,
        obs_start=day,
        cycle_no=1,
        entry_path=dec.entry_path or "S",
        delta_start=dec.p_ep - c,
    )
    basis: dict[str, Any] = {  # v0.7.1 原有字段，顺序不变（V0 回归逐值比较）
        "reasons": list(dec.reasons),
        "level": dec.level.name,
        "d_t": dec.basis["d_t"],
        "a_start": dec.basis["a_start"],
        "p_ep": dec.p_ep,
        "p_ep_day": dec.p_ep_day,
        "l_ep": l_ep,
        "close": c,
        "r2_status": dec.basis["r2_status"],
        "r2_score": dec.basis["r2_score"],
        "v08": dec.basis_v08,  # v0.8 新增数值，输出时单列
    }
    return ep, Event(EventType.START, day, (eid, EventType.START.value, day), dec.transition, basis)


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
