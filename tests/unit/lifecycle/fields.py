"""B-01：逐日全字段断言（13 号第 4 节）。

对夹具的每一天断言：episode_id、等级、cycle_no、obs_start、box_high、H_d、downgraded、obs_issued、
prev_confirm、last_release、dc、rc、xc，以及当日事件及其业务身份。

期望值按"逐日变化 + 延续"书写：spec 只写当天与前一天不同的字段，其余沿用前一天的期望。
解除（REL）或断点（BREAK）的次日起无进行中预警，预警字段自动回到空值（last_release 照常延续）。
表中没有列出的值由规则推导写入，测试中注明"推导"。期望值不得由引擎输出回填。

记法：
- 日期字段（obs_start、last_release）与 episode_id 写标签（episode_id 写启动日的标签），
  也可直接写日期（注入前态时）。
- 等级写 "W"（关注）、"A"（警戒）或 None。
- 事件写缩写元组：("START",)、("ESC",)、("DOWN",)、("REL",)、("BREAK",)、("ISSUE",)、
  ("STAB", 周期)、("OBS", 周期, n)、("REV", 周期, n)、("FAIL", 周期)。
"""

from __future__ import annotations

from typing import Any

from stock_risk.domain.lifecycle.model import EventType, Level

FIELDS = (
    "episode_id",
    "level",
    "cycle_no",
    "obs_start",
    "box_high",
    "h_d",
    "downgraded",
    "obs_issued",
    "prev_confirm",
    "last_release",
    "dc",
    "rc",
    "xc",
)
# 无进行中预警的日子（预警字段为空）
NOEP: dict[str, Any] = {f: None for f in FIELDS if f != "last_release"}
LEVELS = {"W": Level.WATCH, "A": Level.ALERT, None: None}
ABBR = {
    "START": EventType.START,
    "ESC": EventType.ESCALATE,
    "DOWN": EventType.DOWNGRADE,
    "REL": EventType.RELEASE,
    "BREAK": EventType.DATA_BREAK,
    "ISSUE": EventType.DATA_ISSUE,
    "STAB": EventType.STABILIZED,
    "OBS": EventType.RECOVERY_OBS,
    "REV": EventType.RECOVERY_REVOKED,
    "FAIL": EventType.RECOVERY_FAIL,
}


def started(label: str, level: str, **more: Any) -> dict[str, Any]:
    """启动日：新预警，周期 1，obs_start = 当日，计数为 0，其余预警字段清空（第 1.2 节第 5 步）。"""
    return {
        "episode_id": label,
        "level": level,
        "cycle_no": 1,
        "obs_start": label,
        "box_high": None,
        "h_d": None,
        "downgraded": False,
        "obs_issued": False,
        "prev_confirm": False,
        "dc": 0,
        "rc": 0,
        "xc": 0,
        "events": [("START",)],
        **more,
    }


def released(label: str, **more: Any) -> dict[str, Any]:
    """解除日：记录被解除预警的冻结字段；等级、计数、H_d 为空；last_release = 当日（第 1.4 节）。
    prev_confirm 与 obs_issued 为前一日的值（当日不发恢复进度事件、不写入 prev_confirm）。"""
    return {
        "level": None,
        "dc": None,
        "rc": None,
        "xc": None,
        "h_d": None,
        "last_release": label,
        "events": [("REL",)],
        **more,
    }


def reset_at(label: str, cycle: int, **more: Any) -> dict[str, Any]:
    """观察重置（新低、加重或缺价）：obs_start = 当日，cycle_no 加一，box_high、obs_issued 清空，
    三计数清零；当日末尾 prev_confirm 为假（当日不判断确认）。"""
    return {
        "obs_start": label,
        "cycle_no": cycle,
        "box_high": None,
        "obs_issued": False,
        "prev_confirm": False,
        "dc": 0,
        "rc": 0,
        "xc": 0,
        **more,
    }


def check_days(
    run: Any,
    spec: dict[str, dict[str, Any]],
    init: dict[str, Any] | None = None,
    symbol: str = "TEST",
) -> None:
    labels: dict[str, Any] = run.labels
    inv = {d: k for k, d in labels.items()}

    def day_of(v: Any) -> Any:
        return labels[v] if isinstance(v, str) else v

    unknown = set(spec) - set(labels)
    assert not unknown, f"spec 中有不存在的标签: {sorted(unknown)}"
    state: dict[str, Any] = {**NOEP, "last_release": None, **(init or {})}
    problems: list[str] = []
    ended = False
    for r in run.result.results:
        rec, lab = r.record, inv.get(r.record.day, str(r.record.day))
        upd = dict(spec.get(lab, {}))
        events = upd.pop("events", [])
        if ended:
            state.update(NOEP)
        state.update(upd)
        ended = any(e[0] in ("REL", "BREAK") for e in events)
        eid = (
            f"{symbol}:{day_of(state['episode_id']).isoformat()}"
            if state["episode_id"] is not None
            else None
        )
        expect = {
            **state,
            "episode_id": eid,
            "level": LEVELS[state["level"]],
            "obs_start": day_of(state["obs_start"]) if state["obs_start"] else None,
            "last_release": day_of(state["last_release"]) if state["last_release"] else None,
        }
        for f in FIELDS:
            got = getattr(rec, f)
            if got != expect[f]:
                problems.append(f"{lab}.{f}: 期望 {expect[f]!r}，实际 {got!r}")
        want = sorted(_key(e, eid, rec.day, symbol) for e in events)
        have = sorted(e.key for e in r.events)
        if [str(k) for k in want] != [str(k) for k in have]:
            problems.append(f"{lab}.events: 期望 {want}，实际 {have}")
    assert problems == [], "逐日全字段断言不一致：\n" + "\n".join(problems[:40])


def _key(abbr: tuple[Any, ...], eid: str | None, day: Any, symbol: str) -> tuple[Any, ...]:
    kind = ABBR[abbr[0]]
    if kind is EventType.DATA_ISSUE:
        return (kind, symbol, kind.value, day)
    if kind in (
        EventType.START,
        EventType.ESCALATE,
        EventType.DOWNGRADE,
        EventType.RELEASE,
        EventType.DATA_BREAK,
    ):
        return (kind, eid, kind.value, day)
    return (kind, eid, *abbr[1:])
