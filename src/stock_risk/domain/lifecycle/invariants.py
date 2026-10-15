"""13 号文档第 4 节的不变量。①–⑨ 总是检查；v0.8 的 ⑩、⑪ 在相应开关打开时检查。
纯函数，输入一段逐日结果，返回违反项（空表示全部满足）。

夹具测试与以后的真实回放都调用它。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from stock_risk.domain.lifecycle.model import (
    LEVEL_EVENTS,
    RECOVERY_EVENTS,
    DailyRecord,
    DataStatus,
    EndReason,
    EventType,
    Level,
    StepResult,
)
from stock_risk.domain.lifecycle.params import Confirm, EngineParams, ReleaseHyst, ReleasePath
from stock_risk.domain.lifecycle.start_paths import full_alert
from stock_risk.domain.numeric import Kind, lt


@dataclass
class _Confirms:
    """跨日跟踪恢复确认（不变量 6、8）。"""

    open: set[tuple] = field(default_factory=set)  # 已发 RECOVERY_OBS、尚未结束的确认
    issued_cycles: set[tuple] = field(default_factory=set)  # 发过 RECOVERY_OBS 的周期
    failed_cycles: set[tuple] = field(default_factory=set)


def check_invariants(results: Sequence[StepResult], params: EngineParams) -> list[str]:
    out: list[str] = []
    confirms = _Confirms()
    prev: DailyRecord | None = None
    for r in results:
        types = [e.type for e in r.events]
        out += _cycle_and_level(prev, r.record, types)
        out += _box_high(prev, r.record, params)
        out += _counters_and_h_d(r.record, types, params)
        out += _confirm_events(r, confirms)
        out += _confirm_closed(prev, r.record, types)
        out += _evidence_and_basis(r)
        out += _v08_invariants(r, params)
        prev = r.record
    return out


def _same_episode(prev: DailyRecord | None, rec: DailyRecord) -> bool:
    return prev is not None and prev.episode_id == rec.episode_id


def _cycle_and_level(
    prev: DailyRecord | None, rec: DailyRecord, types: list[EventType]
) -> list[str]:
    """1. cycle_no 每日至多加一；每天至多一次等级变化。"""
    out: list[str] = []
    if sum(t in LEVEL_EVENTS for t in types) > 1:
        out.append(f"{rec.day} 不变量1：一天内多次等级变化 {types}")
    if (
        _same_episode(prev, rec)
        and prev is not None
        and rec.cycle_no is not None
        and prev.cycle_no is not None
        and rec.cycle_no - prev.cycle_no > 1
    ):
        out.append(f"{rec.day} 不变量1：cycle_no 一天内加 {rec.cycle_no - prev.cycle_no}")
    return out


def _box_high(prev: DailyRecord | None, rec: DailyRecord, params: EngineParams) -> list[str]:
    """2. box_high 只在 since_obs = K 当天写入，此后到观察重置不变；since_obs ≥ K 时必须有值
    （仅 BOX）。"""
    k = params.candidate.k
    if params.candidate.confirm is not Confirm.BOX or rec.episode_id is None:
        return []
    out: list[str] = []
    if rec.box_high is not None and (rec.since_obs is None or rec.since_obs < k):
        out.append(f"{rec.day} 不变量2：since_obs < K 时 box_high 有值")
    if rec.box_high is None and rec.since_obs is not None and rec.since_obs >= k:
        out.append(f"{rec.day} 不变量2：since_obs ≥ K 时 box_high 为空")
    if prev is not None and _same_episode(prev, rec):
        changed = prev.box_high != rec.box_high and not rec.reset
        if changed and not (prev.box_high is None and rec.since_obs == k):
            out.append(f"{rec.day} 不变量2：box_high 在非企稳首日改变")
    return out


def _counters_and_h_d(rec: DailyRecord, types: list[EventType], params: EngineParams) -> list[str]:
    """3. H_d 只在"关注且 downgraded"时有值，加重当日清空；4. TWO_STAGE 时 xc 恒为 0；
    5. 解除当日没有恢复进度事件；7. 新低日证据为空、强恢复不成立。"""
    out: list[str] = []
    day = rec.day
    if rec.h_d is not None and not (rec.level is Level.WATCH and rec.downgraded):
        out.append(f"{day} 不变量3：H_d 在关注（降级而来）以外有值")
    if EventType.ESCALATE in types and rec.h_d is not None:
        out.append(f"{day} 不变量3：加重当日 H_d 未清空")
    two_stage = params.candidate.release_path is ReleasePath.TWO_STAGE
    if two_stage and ((rec.xc or 0) != 0 or (rec.xc_cand or 0) != 0):
        out.append(f"{day} 不变量4：TWO_STAGE 下 xc 不为 0")
    if EventType.RELEASE in types and any(t in RECOVERY_EVENTS for t in types):
        out.append(f"{day} 不变量5：解除当日有恢复进度事件 {types}")
    if rec.new_low and (rec.evidence or rec.strong):
        out.append(f"{day} 不变量7：新低日证据非空或强恢复成立")
    return out


def _confirm_events(r: StepResult, confirms: _Confirms) -> list[str]:
    """6. 缺价日末尾 prev_confirm 为假；每次确认至多以一个事件结束。
    8. 每周期至多一次恢复失败，且只在该周期发过恢复观察时出现。"""
    out: list[str] = []
    rec, day = r.record, r.record.day
    if rec.data_status is DataStatus.MISSING and rec.prev_confirm:
        out.append(f"{day} 不变量6：缺价日末尾 prev_confirm 为真")
    for e in r.events:
        if e.type is EventType.RECOVERY_OBS:
            confirms.open.add(e.identity)
            confirms.issued_cycles.add(e.identity[:2])
        elif e.type is EventType.RECOVERY_REVOKED:
            if e.identity not in confirms.open:
                out.append(f"{day} 不变量6：撤销了不存在或已结束的确认 {e.identity}")
            confirms.open.discard(e.identity)
        elif e.type is EventType.RECOVERY_FAIL:
            out += _recovery_fail(day, e.identity[:2], confirms)
    types = {e.type for e in r.events}
    if EventType.RECOVERY_FAIL in types and EventType.RECOVERY_REVOKED in types:
        out.append(f"{day} 不变量6：同日既撤销又恢复失败")
    return out


def _recovery_fail(day: object, ep_cycle: tuple, confirms: _Confirms) -> list[str]:
    out: list[str] = []
    closing = {c for c in confirms.open if c[:2] == ep_cycle}
    if len(closing) > 1:
        out.append(f"{day} 不变量6：同一周期有多个未结束的确认")
    confirms.open -= closing
    if ep_cycle in confirms.failed_cycles:
        out.append(f"{day} 不变量8：周期 {ep_cycle} 重复发出 RECOVERY_FAIL")
    if ep_cycle not in confirms.issued_cycles:
        out.append(f"{day} 不变量8：周期 {ep_cycle} 未发恢复观察却发出 RECOVERY_FAIL")
    confirms.failed_cycles.add(ep_cycle)
    return out


def _confirm_closed(
    prev: DailyRecord | None, rec: DailyRecord, types: list[EventType]
) -> list[str]:
    """6.（T08a B-02）上一交易日确认成立、当日未解除时，若当日确认消失或发生观察重置，
    当日必须有终止该确认的事件（RECOVERY_REVOKED 或 RECOVERY_FAIL）。"""
    if prev is None or not _same_episode(prev, rec) or rec.episode_id is None:
        return []
    if prev.end_reason is not None or not prev.prev_confirm:
        return []
    if rec.end_reason is EndReason.RELEASE:
        return []
    if rec.confirmed is True and not rec.reset:
        return []
    if EventType.RECOVERY_REVOKED in types or EventType.RECOVERY_FAIL in types:
        return []
    return [f"{rec.day} 不变量6：上一交易日的确认在当日消失或被重置，但没有终止事件"]


def _evidence_and_basis(r: StepResult) -> list[str]:
    """9. 事件依据中的计数是今日候选计数。"""
    out: list[str] = []
    rec = r.record
    for e in r.events:
        if e.type is EventType.DOWNGRADE and e.basis.get("dc_cand") != rec.dc_cand:
            out.append(f"{rec.day} 不变量9：DOWNGRADE 依据计数与候选计数不同")
        if e.type is EventType.RELEASE:
            for key in ("rc_cand", "xc_cand", "dc_cand"):
                if key in e.basis and e.basis[key] != getattr(rec, key):
                    out.append(f"{rec.day} 不变量9：RELEASE 依据 {key} 与候选计数不同")
    return out


def _v08_invariants(r: StepResult, params: EngineParams) -> list[str]:
    """⑩ t11_full = ON 时，每次进入警戒当日完整警戒条件成立；
    ⑪ release_hyst = EFFECTIVE 时，启动当日 a′ 不成立（10 号第 1A.4、1A.5 节）。"""
    out: list[str] = []
    rec, sw = r.record, params.switches
    types = {e.type for e in r.events}
    entering = EventType.ESCALATE in types or (
        EventType.START in types and rec.level is Level.ALERT
    )
    ok = rec.d_ep is not None and rec.dd_ep is not None and full_alert(rec.d_ep, rec.dd_ep, params)
    if sw.t11_full and entering and not ok:
        out.append(f"{rec.day} 不变量10：进入警戒当日完整警戒条件不成立")
    if sw.release_hyst is ReleaseHyst.EFFECTIVE and EventType.START in types:
        assert rec.p_ep is not None and rec.atr_ep is not None and rec.delta_start is not None
        if rec.entry_path == "L":
            e_entry = sw.x_long * rec.p_ep
        else:
            e_entry = max(params.candidate.k1 * rec.atr_ep, sw.x1 * rec.p_ep)
        th = min(e_entry, rec.delta_start)
        if lt(rec.delta_start, params.assumptions.h * th, Kind.PRICE, params.tol):
            out.append(f"{rec.day} 不变量11：启动当日 a′ 已成立")
    return out
