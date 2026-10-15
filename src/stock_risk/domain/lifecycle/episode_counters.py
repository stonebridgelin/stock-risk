"""第 10 步：计数四阶段（E-01）、降级、解除与解除原因（E-03、I21）。

从 episode_step 拆出（T08a D-03）。
"""

from __future__ import annotations

from dataclasses import replace

from stock_risk.domain.lifecycle.episode_day import EpisodeDay, WorkDay
from stock_risk.domain.lifecycle.episode_rules import ordered
from stock_risk.domain.lifecycle.model import DIRECT_STRONG_RECOVERY, Event, EventType, Level
from stock_risk.domain.lifecycle.params import EngineParams, ReleasePath


def counters(w: WorkDay, params: EngineParams) -> EpisodeDay | None:
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
    w: WorkDay,
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


def _release(
    w: WorkDay,
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
