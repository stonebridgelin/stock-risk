"""四种输出（events、daily、episodes、图表）之间的一致性检查（T04b）。纯函数，读回写出的文件后检查。

从 replay_outputs 拆出（T08a D-03）。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from stock_risk.domain.lifecycle.model import DataStatus


def check_consistency(
    events: Sequence[Mapping[str, Any]],
    daily: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    chart_events: Iterable[tuple[str, str]],
) -> list[str]:
    """同一事件在 events、daily、episodes 与图表中的日期一致（读回写出的文件后检查）。

    chart_events：从 chart.html 读回的（日期, 事件类型）。
    """
    day_rows = {r["day"]: r for r in daily}
    problems: list[str] = []
    if sorted((r["date"], r["event_type"]) for r in events) != sorted(chart_events):
        problems.append("图表中的事件标记与 events.csv 不一致")
    problems += _events_vs_daily(events, day_rows)
    level_days = {(r["date"], r["event_type"]) for r in events}
    for d in daily:
        if d["transition"] in ("T6", "T7") and (d["day"], "START") not in level_days:
            problems.append(f"{d['day']}：daily.csv 有启动转移但 events.csv 没有 START")
    for ep in episodes:
        problems += _episode_vs_others(ep, events, day_rows)
    return problems


_EXPECT = {
    "START": lambda d: d["transition"] in ("T6", "T7"),
    "ESCALATE": lambda d: d["transition"] in ("T9", "T10", "T11"),
    "DOWNGRADE": lambda d: d["transition"] == "T15",
    "RELEASE": lambda d: d["end_reason"] == "RELEASE",
    "DATA_BREAK": lambda d: d["end_reason"] == "DATA_BREAK",
    "DATA_ISSUE": lambda d: (
        d["data_status"] in (DataStatus.MISSING.value, DataStatus.DATA_BREAK.value)
    ),
}


def _events_vs_daily(
    events: Sequence[Mapping[str, Any]], day_rows: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """每个事件在 daily.csv 当日有对应记录，episode_id 一致。"""
    problems: list[str] = []
    for r in events:
        d = day_rows.get(r["date"])
        if d is None:
            problems.append(f"{r['date']} {r['event_type']}：daily.csv 中没有这一天")
            continue
        check = _EXPECT.get(r["event_type"])
        if check is not None and not check(d):
            problems.append(f"{r['date']} {r['event_type']}：与 daily.csv 当日记录不符")
        if (
            r["episode_id"]
            and r["event_type"] != "DATA_BREAK"
            and d["episode_id"] != r["episode_id"]
        ):
            problems.append(f"{r['date']} {r['event_type']}：daily.csv 的 episode_id 不同")
    return problems


def _episode_vs_others(
    ep: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    day_rows: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """episodes 的启动日、降级日、终止日与 events.csv 一致，截尾日与 daily.csv 一致。"""
    problems: list[str] = []
    eid = ep["episode_id"]
    own = {(r["event_type"], r["date"]) for r in events if r["episode_id"] == eid}
    if ("START", ep["start_date"]) not in own:
        problems.append(f"{eid}：episodes 的启动日与 events.csv 不一致")
    if ep["downgrade_date"] and ("DOWNGRADE", ep["downgrade_date"]) not in own:
        problems.append(f"{eid}：episodes 的降级日与 events.csv 不一致")
    if (
        ep["end_reason"] in ("RELEASE", "DATA_BREAK")
        and (ep["end_reason"], ep["end_date"]) not in own
    ):
        problems.append(f"{eid}：episodes 的终止日与 events.csv 不一致")
    if (
        ep["end_reason"] in ("LISTING_END", "EXPLORATION_END")
        and day_rows.get(ep["end_date"], {}).get("end_reason") != ep["end_reason"]
    ):
        problems.append(f"{eid}：截尾日与 daily.csv 不一致")
    return problems
