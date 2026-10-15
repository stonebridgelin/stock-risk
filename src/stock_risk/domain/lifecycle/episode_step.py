"""有预警时的每日处理（10 号第 1.2 节 C，第 6–11 步；第 1.4、1.5 节；勘误 E-01、E-03）。

第 10 步计数与解除在 episode_counters.py（T08a D-03 拆分）。

一日的计算放在局部工作对象 WorkDay 中逐步推进；输入的 Episode 不被修改，输出新的 Episode。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from stock_risk.domain.indicator_source import IndicatorSource
from stock_risk.domain.lifecycle.episode_counters import counters
from stock_risk.domain.lifecycle.episode_day import EpisodeDay, WorkDay
from stock_risk.domain.lifecycle.episode_rules import (
    REVOKE_REASON,
    a_prime,
    confirmation,
    observation_reset,
    theta,
)
from stock_risk.domain.lifecycle.model import (
    EVID_A,
    EVID_A1,
    EVID_B,
    EVID_B1,
    EVID_C,
    Episode,
    Event,
    EventType,
    Level,
    ResetReason,
)
from stock_risk.domain.lifecycle.params import Confirm, EngineParams, RecScale
from stock_risk.domain.lifecycle.start_paths import full_alert
from stock_risk.domain.numeric import Kind, ge, lt


def step_episode(ep: Episode, day: date, src: IndicatorSource, params: EngineParams) -> EpisodeDay:
    w = _transition_evidence(ep, day, src, params)
    esc = _escalation(w, params)
    if esc is not None:
        _escalate(w, *esc)
    else:
        _observation_evidence(w, src, params)
        released = counters(w, params)
        if released is not None:
            return released
    _progress_events(w, params)
    w.ep = replace(w.ep, prev_confirm=w.confirm)
    w.fields["reset_reasons"] = tuple(w.reasons)
    return EpisodeDay(w.ep, tuple(w.events), w.fields, w.transition, released=False)


def _transition_evidence(
    ep: Episode, day: date, src: IndicatorSource, params: EngineParams
) -> WorkDay:
    """第 6 步（用昨日字段与今日收盘）与第 7 步（新低：更新 L_ep，观察重置）。"""
    c = src.close(day)
    new_low = lt(c, ep.l_ep, Kind.PRICE, params.tol)
    cur, giveback = ep, None
    if cur.level is Level.WATCH and cur.downgraded:
        assert cur.h_d is not None
        h_d = max(cur.h_d, c)
        cur, giveback = replace(cur, h_d=h_d), (h_d - c) / cur.atr_ep
    d_ep, dd_ep = (ep.p_ep - c) / ep.atr_ep, 1 - c / ep.p_ep
    w = WorkDay(cur, day, c, d_ep, dd_ep, new_low, giveback, snapshot=ep)
    if new_low:
        w.ep = observation_reset(replace(w.ep, l_ep=c, l_ep_day=day), day)
        w.reasons.append(ResetReason.NEW_LOW)
    w.fields.update(
        d_ep=w.d_ep,
        dd_ep=w.dd_ep,
        theta=theta(ep, params),
        new_low=new_low,
        giveback=giveback,
        confirmed=False,
        stable=False,
        long_stable=False,
        strong=False,
        evidence=frozenset(),
    )
    return w


def _escalation(w: WorkDay, params: EngineParams) -> tuple[str, str] | None:
    """第 8 步：加重组（仅关注），自上而下取首条命中（T9–T11）。"""
    cand, tol, ep = params.candidate, params.tol, w.ep
    if ep.level is not Level.WATCH:
        return None
    full = full_alert(w.d_ep, w.dd_ep, params)  # V0 下即 D_ep ≥ k₂
    if not ep.downgraded and full:
        return "T9", "D_EP_GE_K2"
    if ep.downgraded and w.new_low:
        return "T10", "NEW_LOW"
    t11_ok = full or not params.switches.t11_full  # t11_full = ON：还要完整警戒条件
    if (
        ep.downgraded
        and w.giveback is not None
        and ge(w.giveback, cand.k1, Kind.RATIO, tol)
        and t11_ok
    ):
        return "T11", "GIVEBACK"
    return None


def _escalate(w: WorkDay, transition: str, why: str) -> None:
    """加重：升为警戒，观察重置（同日至多一次），三计数清零，H_d 清空，跳到第 11 步。"""
    h_d_before = w.ep.h_d
    if not w.reasons:
        w.ep = observation_reset(w.ep, w.day)
    w.ep = replace(
        w.ep,
        level=Level.ALERT,
        h_d=None,
        dc=0,
        rc=0,
        xc=0,
        dc_segment=(),
        rc_segment=(),
        xc_segment=(),
    )
    w.reasons.append(ResetReason.ESCALATE)
    w.escalated, w.transition = True, transition
    basis = {"reason": why, "d_ep": w.d_ep, "giveback": w.giveback, "h_d": h_d_before}
    key = (w.ep.episode_id, EventType.ESCALATE.value, w.day)
    w.events.append(Event(EventType.ESCALATE, w.day, key, transition, {**basis, "close": w.close}))
    w.fields.update(dc_cand=0, rc_cand=None, xc_cand=0)


def _observation_evidence(w: WorkDay, src: IndicatorSource, params: EngineParams) -> None:
    """第 9 步：since_obs、企稳、长期稳定、恢复确认、U、强恢复、E_down / E_rel（第 1.3 节）。

    用第 7 步之后的字段；今日新低时一律为假或空。
    """
    cand, tol, c = params.candidate, params.tol, w.close
    if not w.new_low:
        since = w.ep.since_obs + 1
        run_high = c if w.ep.run_high is None else max(w.ep.run_high, c)
        box_high = w.ep.box_high
        if since == cand.k and cand.confirm is Confirm.BOX:
            box_high = run_high  # obs_start 后第 1 至第 K 个有效交易日的最高收盘，在第 K 天冻结
        w.ep = replace(w.ep, since_obs=since, run_high=run_high, box_high=box_high)
        w.confirm, line = confirmation(w.ep, w.day, c, src, params)
        w.fields["confirm_line"] = line
    ep = w.ep
    stable = not w.new_low and ep.since_obs >= cand.k
    long_stable = not w.new_low and ep.since_obs >= params.assumptions.k_stab
    sw = params.switches
    # rec_scale = PRICE：恢复侧用 ATR_rec = ATR_ep × L_ep / P_ep（L_ep 为第 7 步之后的值）
    rec_atr = ep.atr_ep * ep.l_ep / ep.p_ep if sw.rec_scale is RecScale.PRICE else ep.atr_ep
    amount = min(cand.u_rel * rec_atr, params.assumptions.phi * (ep.p_ep - ep.l_ep))
    strong = w.confirm and ge(c - ep.l_ep, amount, Kind.PRICE, tol)
    evidence: set[str] = set()
    if not w.new_low and ep.level is Level.ALERT:
        a_pct = sw.x2 > 0 and lt(w.dd_ep, sw.x2, Kind.RATIO, tol)
        if lt(w.d_ep, cand.k2, Kind.RATIO, tol) or a_pct:
            evidence.add(EVID_A)
        if w.confirm:
            evidence.add(EVID_B)
        if long_stable:
            evidence.add(EVID_C)
    elif not w.new_low:
        if a_prime(ep, c, w.d_ep, params):
            evidence.add(EVID_A1)
        if strong:
            evidence.add(EVID_B1)
    w.fields.update(
        confirmed=w.confirm,
        stable=stable,
        long_stable=long_stable,
        u=(c - ep.l_ep) / rec_atr,
        atr_rec=rec_atr if sw.rec_scale is RecScale.PRICE else None,
        strong_threshold=ep.l_ep + amount,
        strong=strong,
        evidence=frozenset(evidence),
    )


def _progress_events(w: WorkDay, params: EngineParams) -> None:
    """第 11 步：恢复进度事件（第 1.5 节），按顺序判断。"""
    snap, eid, c = w.snapshot, w.ep.episode_id, w.close
    if w.new_low and snap.obs_issued:  # 被终止的周期发出过恢复观察
        basis = {"reason": "NEW_LOW", "close": c, "confirm_open": snap.prev_confirm}
        w.events.append(Event(EventType.RECOVERY_FAIL, w.day, (eid, snap.cycle_no), None, basis))
    elif w.escalated and not w.new_low and snap.prev_confirm:
        key = (eid, snap.cycle_no, snap.obs_n)
        w.events.append(
            Event(EventType.RECOVERY_REVOKED, w.day, key, None, {"reason": "ESCALATED", "close": c})
        )
    if w.reasons:  # 当日有观察重置，不判断企稳与确认的变化
        return
    ep, line = w.ep, w.fields.get("confirm_line")
    if ep.since_obs == params.candidate.k and not ep.stabilized:
        w.events.append(Event(EventType.STABILIZED, w.day, (eid, ep.cycle_no), None, {"close": c}))
        ep = replace(ep, stabilized=True)
    if w.confirm and not ep.prev_confirm:
        n = ep.obs_n + 1
        basis = {"close": c, "line": line, "l_ep": ep.l_ep, "l_ep_day": ep.l_ep_day}
        w.events.append(Event(EventType.RECOVERY_OBS, w.day, (eid, ep.cycle_no, n), None, basis))
        ep = replace(ep, obs_n=n)
    elif not w.confirm and ep.prev_confirm:
        basis = {"reason": REVOKE_REASON[params.candidate.confirm], "close": c, "line": line}
        key = (eid, ep.cycle_no, ep.obs_n)
        w.events.append(Event(EventType.RECOVERY_REVOKED, w.day, key, None, basis))
    w.ep = ep
