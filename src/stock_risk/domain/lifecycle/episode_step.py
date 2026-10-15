"""有预警时的每日处理（10 号第 1.2 节 C，第 6–11 步；第 1.4、1.5 节；勘误 E-01、E-03）。

一日的计算放在局部工作对象 _Day 中逐步推进；输入的 Episode 不被修改，输出新的 Episode。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

from stock_risk.domain.indicator_source import IndicatorSource
from stock_risk.domain.lifecycle.episode_rules import (
    REVOKE_REASON,
    a_prime,
    confirmation,
    observation_reset,
    ordered,
    theta,
)
from stock_risk.domain.lifecycle.model import (
    DIRECT_STRONG_RECOVERY,
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
from stock_risk.domain.lifecycle.params import Confirm, EngineParams, RecScale, ReleasePath
from stock_risk.domain.lifecycle.start_paths import full_alert
from stock_risk.domain.numeric import Kind, ge, lt


@dataclass(frozen=True, slots=True)
class EpisodeDay:
    """C 段一日的结果。released 为真时 episode 为当日被解除的预警。fields 写入诊断记录。"""

    episode: Episode
    events: tuple[Event, ...]
    fields: dict[str, Any]
    transition: str | None
    released: bool


@dataclass(slots=True)
class _Day:
    """一日的工作状态（仅在 step_episode 内部使用）。"""

    ep: Episode
    day: date
    close: float
    d_ep: float
    dd_ep: float
    new_low: bool
    giveback: float | None
    snapshot: Episode  # 观察重置前（昨日日末）的预警，用于终止事件的身份
    reasons: list[ResetReason] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)
    confirm: bool = False
    escalated: bool = False
    transition: str | None = None


def step_episode(ep: Episode, day: date, src: IndicatorSource, params: EngineParams) -> EpisodeDay:
    w = _transition_evidence(ep, day, src, params)
    esc = _escalation(w, params)
    if esc is not None:
        _escalate(w, *esc)
    else:
        _observation_evidence(w, src, params)
        released = _counters(w, params)
        if released is not None:
            return released
    _progress_events(w, params)
    w.ep = replace(w.ep, prev_confirm=w.confirm)
    w.fields["reset_reasons"] = tuple(w.reasons)
    return EpisodeDay(w.ep, tuple(w.events), w.fields, w.transition, released=False)


def _transition_evidence(
    ep: Episode, day: date, src: IndicatorSource, params: EngineParams
) -> _Day:
    """第 6 步（用昨日字段与今日收盘）与第 7 步（新低：更新 L_ep，观察重置）。"""
    c = src.close(day)
    new_low = lt(c, ep.l_ep, Kind.PRICE, params.tol)
    cur, giveback = ep, None
    if cur.level is Level.WATCH and cur.downgraded:
        assert cur.h_d is not None
        h_d = max(cur.h_d, c)
        cur, giveback = replace(cur, h_d=h_d), (h_d - c) / cur.atr_ep
    d_ep, dd_ep = (ep.p_ep - c) / ep.atr_ep, 1 - c / ep.p_ep
    w = _Day(cur, day, c, d_ep, dd_ep, new_low, giveback, snapshot=ep)
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


def _escalation(w: _Day, params: EngineParams) -> tuple[str, str] | None:
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


def _escalate(w: _Day, transition: str, why: str) -> None:
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


def _observation_evidence(w: _Day, src: IndicatorSource, params: EngineParams) -> None:
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


def _counters(w: _Day, params: EngineParams) -> EpisodeDay | None:
    """第 10 步：计数四阶段（E-01）。

    昨日日末计数 → 今日候选计数 → 用候选计数判断转移 → 日末计数。
    """
    cand, ep = params.candidate, w.ep
    ev: frozenset[str] = w.fields["evidence"]
    direct = cand.release_path is ReleasePath.DIRECT
    xc_cand = (ep.xc + 1 if w.fields["strong"] else 0) if direct else 0
    # xc 计数段内每日的证据集合（T08a B-06，只用于 RELEASE 依据）
    xseg = (*ep.xc_segment, ev) if xc_cand else ()
    if ep.level is Level.WATCH:
        return _counters_watch(w, params, ev, xc_cand, xseg)
    dc_cand = ep.dc + 1 if ev else 0
    w.fields.update(dc_cand=dc_cand, rc_cand=None, xc_cand=xc_cand)
    counts = {"dc_cand": dc_cand, "xc_cand": xc_cand}
    if direct and xc_cand == cand.d:  # T14
        return _release(w, "T14", [DIRECT_STRONG_RECOVERY], (), counts, xseg)
    if dc_cand != cand.d:  # T16
        w.transition = "T16"
        dseg = (*ep.dc_segment, ev) if ev else ()
        w.ep = replace(ep, dc=dc_cand, xc=xc_cand, dc_segment=dseg, xc_segment=xseg)
        return None
    # T15：降为关注；downgraded = 是；H_d = C；dc、rc 清零，xc 保留
    segment = [ordered(s) for s in (*ep.dc_segment, ev)]
    w.transition = "T15"
    basis = {"reasons": ordered(ev), **counts, "segment": segment, "close": w.close}
    key = (ep.episode_id, EventType.DOWNGRADE.value, w.day)
    w.events.append(Event(EventType.DOWNGRADE, w.day, key, "T15", basis))
    w.ep = replace(
        ep,
        level=Level.WATCH,
        downgraded=True,
        h_d=w.close,
        dc=0,
        rc=0,
        xc=xc_cand,
        dc_segment=(),
        rc_segment=(),
        xc_segment=xseg,
    )
    return None


def _counters_watch(
    w: _Day,
    params: EngineParams,
    ev: frozenset[str],
    xc_cand: int,
    xseg: tuple[frozenset[str], ...],
) -> EpisodeDay | None:
    """关注时：xc 或 rc 的今日候选计数达到 d → 解除（T12），原因按 E-03；否则 T13。

    只由 xc 达到 d 时，rc 计数段（segment）写空，xc 计数段另列（T08a B-06）。"""
    cand, ep = params.candidate, w.ep
    rc_cand = ep.rc + 1 if ev else 0
    w.fields.update(dc_cand=None, rc_cand=rc_cand, xc_cand=xc_cand)
    hit_x = cand.release_path is ReleasePath.DIRECT and xc_cand == cand.d
    hit_r = rc_cand == cand.d
    if hit_x or hit_r:
        segment = (*ep.rc_segment, ev) if hit_r else ()
        why = [DIRECT_STRONG_RECOVERY] if hit_x else []
        if hit_r:  # I21：rc 达到 d 时记达到 d 当日的证据（a′、b′ 排序）；计数段每日证据另存
            why += ordered(ev)
        counts = {"rc_cand": rc_cand, "xc_cand": xc_cand}
        direct = cand.release_path is ReleasePath.DIRECT
        return _release(w, "T12", why, segment, counts, xseg if direct else None)
    w.transition = "T13"
    rseg = (*ep.rc_segment, ev) if ev else ()
    w.ep = replace(ep, rc=rc_cand, xc=xc_cand, rc_segment=rseg, xc_segment=xseg)
    return None


def _progress_events(w: _Day, params: EngineParams) -> None:
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


def _release(
    w: _Day,
    transition: str,
    reasons: list[str],
    rc_segment: tuple[frozenset[str], ...],
    counts: dict[str, int],
    xc_segment: tuple[frozenset[str], ...] | None,
) -> EpisodeDay:
    """解除：预警结束；当日不发恢复进度事件（第 1.5 节）。

    xc_segment：DIRECT 时的 xc 计数段（每日证据），TWO_STAGE 时为 None（依据中不出现）。"""
    basis = {
        "reasons": reasons,
        "today_evidence": ordered(w.fields["evidence"]),
        "strong": w.fields["strong"],
        "segment": [ordered(s) for s in rc_segment],
        "close": w.close,
        **counts,
    }
    if xc_segment is not None:
        basis["xc_segment"] = [ordered(s) for s in xc_segment]
    key = (w.ep.episode_id, EventType.RELEASE.value, w.day)
    w.events.append(Event(EventType.RELEASE, w.day, key, transition, basis))
    w.fields["reset_reasons"] = tuple(w.reasons)
    return EpisodeDay(w.ep, tuple(w.events), w.fields, transition, released=True)
