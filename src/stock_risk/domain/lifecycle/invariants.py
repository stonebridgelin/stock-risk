"""13 号文档第 4 节的不变量。①–⑨ 总是检查；v0.8 的 ⑩、⑪ 在相应开关打开时检查。
纯函数，输入一段逐日结果，返回违反项（空表示全部满足）。

夹具测试与以后的真实回放都调用它。
"""

from __future__ import annotations

from collections.abc import Sequence

from stock_risk.domain.lifecycle.model import (
    LEVEL_EVENTS,
    RECOVERY_EVENTS,
    DataStatus,
    EventType,
    Level,
    StepResult,
)
from stock_risk.domain.lifecycle.params import Confirm, EngineParams, ReleaseHyst, ReleasePath
from stock_risk.domain.lifecycle.start_paths import full_alert
from stock_risk.domain.numeric import Kind, lt


def check_invariants(results: Sequence[StepResult], params: EngineParams) -> list[str]:
    out: list[str] = []
    k = params.candidate.k
    open_confirms: set[tuple] = set()  # 已发 RECOVERY_OBS、尚未结束的确认 (episode, cycle, n)
    issued_cycles: set[tuple] = set()  # 发过 RECOVERY_OBS 的 (episode, cycle)
    failed_cycles: set[tuple] = set()
    prev = None
    for r in results:
        rec, day = r.record, r.record.day
        types = [e.type for e in r.events]

        # 1. cycle_no 每日至多加一；每天至多一次等级变化
        if sum(t in LEVEL_EVENTS for t in types) > 1:
            out.append(f"{day} 不变量1：一天内多次等级变化 {types}")
        same_episode = prev is not None and prev.episode_id == rec.episode_id
        if (
            same_episode
            and rec.cycle_no is not None
            and prev.cycle_no is not None
            and rec.cycle_no - prev.cycle_no > 1
        ):
            out.append(f"{day} 不变量1：cycle_no 一天内加 {rec.cycle_no - prev.cycle_no}")

        # 2. box_high 只在 since_obs = K 当天写入，此后到观察重置不变（仅 BOX）
        if params.candidate.confirm is Confirm.BOX and rec.episode_id is not None:
            if rec.box_high is not None and (rec.since_obs is None or rec.since_obs < k):
                out.append(f"{day} 不变量2：since_obs < K 时 box_high 有值")
            changed = same_episode and prev.box_high != rec.box_high and not rec.reset
            if changed and not (prev.box_high is None and rec.since_obs == k):
                out.append(f"{day} 不变量2：box_high 在非企稳首日改变")

        # 3. H_d 只在"关注且 downgraded"时有值；加重当日清空
        if rec.h_d is not None and not (rec.level is Level.WATCH and rec.downgraded):
            out.append(f"{day} 不变量3：H_d 在关注（降级而来）以外有值")
        if EventType.ESCALATE in types and rec.h_d is not None:
            out.append(f"{day} 不变量3：加重当日 H_d 未清空")

        # 4. TWO_STAGE 时 xc 恒为 0
        two_stage = params.candidate.release_path is ReleasePath.TWO_STAGE
        if two_stage and ((rec.xc or 0) != 0 or (rec.xc_cand or 0) != 0):
            out.append(f"{day} 不变量4：TWO_STAGE 下 xc 不为 0")

        # 5. 解除当日没有恢复进度事件
        if EventType.RELEASE in types and any(t in RECOVERY_EVENTS for t in types):
            out.append(f"{day} 不变量5：解除当日有恢复进度事件 {types}")

        # 6. 缺价日末尾 prev_confirm 为假；每次确认至多以一个事件结束
        if rec.data_status is DataStatus.MISSING and rec.prev_confirm:
            out.append(f"{day} 不变量6：缺价日末尾 prev_confirm 为真")
        for e in r.events:
            if e.type is EventType.RECOVERY_OBS:
                open_confirms.add(e.identity)
                issued_cycles.add(e.identity[:2])
            elif e.type is EventType.RECOVERY_REVOKED:
                if e.identity not in open_confirms:
                    out.append(f"{day} 不变量6：撤销了不存在或已结束的确认 {e.identity}")
                open_confirms.discard(e.identity)
            elif e.type is EventType.RECOVERY_FAIL:
                ep_cycle = e.identity[:2]
                closing = {c for c in open_confirms if c[:2] == ep_cycle}
                if len(closing) > 1:
                    out.append(f"{day} 不变量6：同一周期有多个未结束的确认")
                open_confirms -= closing
                # 8. 每周期至多一次恢复失败，且只在该周期发过恢复观察时出现
                if ep_cycle in failed_cycles:
                    out.append(f"{day} 不变量8：周期 {ep_cycle} 重复发出 RECOVERY_FAIL")
                if ep_cycle not in issued_cycles:
                    out.append(f"{day} 不变量8：周期 {ep_cycle} 未发恢复观察却发出 RECOVERY_FAIL")
                failed_cycles.add(ep_cycle)
        if EventType.RECOVERY_FAIL in types and EventType.RECOVERY_REVOKED in types:
            out.append(f"{day} 不变量6：同日既撤销又恢复失败")

        # 7. 新低日 E_down、E_rel 为空，强恢复不成立
        if rec.new_low and (rec.evidence or rec.strong):
            out.append(f"{day} 不变量7：新低日证据非空或强恢复成立")

        # 9. 事件依据中的计数是今日候选计数
        for e in r.events:
            if e.type is EventType.DOWNGRADE and e.basis.get("dc_cand") != rec.dc_cand:
                out.append(f"{day} 不变量9：DOWNGRADE 依据计数与候选计数不同")
            if e.type is EventType.RELEASE:
                for key in ("rc_cand", "xc_cand", "dc_cand"):
                    if key in e.basis and e.basis[key] != getattr(rec, key):
                        out.append(f"{day} 不变量9：RELEASE 依据 {key} 与候选计数不同")
        out += _v08_invariants(r, params)
        prev = rec
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
