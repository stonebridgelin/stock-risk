"""回放结果 → 输出行（events、daily、episodes）与四种输出之间的一致性检查（T04b）。纯函数。"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from datetime import date
from typing import Any

from stock_risk.application.replay import SymbolReplay
from stock_risk.domain.lifecycle.model import (
    DailyRecord,
    DataStatus,
    EndReason,
    Event,
    EventType,
    Level,
)

DAILY_FIELDS = [f.name for f in fields(DailyRecord)]
EVENT_COLUMNS = [
    "run_id",
    "candidate_id",
    "target",
    "date",
    "event_type",
    "identity",
    "episode_id",
    "level_before",
    "level_after",
    "transition",
    "reasons",
    "evidence",
    "segment",
    "values",
    "values_v08",  # v0.8 新增的 START 数值（第 1A.3 节）；原有列不变
]
EPISODE_COLUMNS = [
    "run_id",
    "candidate_id",
    "target",
    "episode_id",
    "start_date",
    "start_close",
    "start_reasons",
    "initial_level",
    "low_date",
    "low_close",
    "first_stabilized_date",
    "first_stabilized_close",
    "first_obs_date",
    "first_obs_close",
    "downgrade_date",
    "downgrade_close",
    "end_date",
    "end_close",
    "end_reason",
    "n_recovery_obs",
    "n_revoked",
    "n_recovery_fail",
    "n_escalate",
    "duration_trading_days",
    "observed_valid_days",
    # v0.8 新增（第 1A.3、1A.7 节）
    "entry_path",
    "delta_start",
    "prev_episode_id",
    "days_since_prev_release",
    "start_below_prev_l_ep",
]


def _jsonable(v: Any) -> Any:
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, (frozenset, set, tuple, list)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, Level):
        return v.name
    return v


def _cell(v: Any) -> Any:
    """CSV 单元格：集合与元组写成 JSON，枚举写名称。"""
    if v is None:
        return ""
    if isinstance(v, Level):
        return v.name
    if isinstance(v, (frozenset, set)):
        return json.dumps(sorted(v), ensure_ascii=False)
    if isinstance(v, (tuple, list)):
        return json.dumps(_jsonable(list(v)), ensure_ascii=False)
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, bool):
        return "true" if v else "false"
    return v.value if hasattr(v, "value") else v


def closes_by_day(rep: SymbolReplay) -> dict[date, float]:
    return {b.day: b.adj_close for b in rep.prepared.data.history.bars}


def daily_rows(rep: SymbolReplay, run_id: str) -> list[dict[str, Any]]:
    """引擎诊断记录原样写出，另加 run_id、候选编号、对象与当日复权收盘。"""
    closes = closes_by_day(rep)
    out = []
    for r in rep.result.results:
        rec = asdict(r.record)
        row = {"run_id": run_id, "candidate_id": rep.candidate_id, "target": rep.target.label}
        row.update({k: _cell(rec[k]) for k in DAILY_FIELDS})
        row["close"] = _cell(closes.get(r.record.day))
        out.append(row)
    return out


def _levels(rep: SymbolReplay) -> dict[date, tuple[Level | None, Level | None]]:
    out, prev = {}, None
    for r in rep.result.results:
        out[r.record.day] = (prev, r.record.level)
        prev = r.record.level
    return out


def event_rows(rep: SymbolReplay, run_id: str) -> list[dict[str, Any]]:
    levels = _levels(rep)
    recs = {r.record.day: r.record for r in rep.result.results}
    out = []
    for e in rep.result.events:
        basis = dict(e.basis)
        reasons = basis.pop("reasons", None) or ([basis["reason"]] if "reason" in basis else [])
        basis.pop("reason", None)
        today = basis.pop("today_evidence", None)
        segment = basis.pop("segment", None)
        v08 = basis.pop("v08", None)
        before, after = levels[e.day]
        out.append(
            {
                "run_id": run_id,
                "candidate_id": rep.candidate_id,
                "target": rep.target.label,
                "date": e.day.isoformat(),
                "event_type": e.type.value,
                "identity": json.dumps(_jsonable(list(e.identity)), ensure_ascii=False),
                "episode_id": "" if e.type is EventType.DATA_ISSUE else e.identity[0],
                "level_before": _cell(before),
                "level_after": _cell(after),
                "transition": e.transition or "",
                "reasons": json.dumps(reasons, ensure_ascii=False),
                "evidence": json.dumps(
                    today if today is not None else sorted(recs[e.day].evidence), ensure_ascii=False
                ),
                "segment": "" if segment is None else json.dumps(segment, ensure_ascii=False),
                "values": json.dumps(_jsonable(basis), ensure_ascii=False),
                "values_v08": "" if v08 is None else json.dumps(_jsonable(v08), ensure_ascii=False),
            }
        )
    return out


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
    results = rep.result.results
    days = [r.record.day for r in results]
    by_eid: dict[str, list[Event]] = {}
    for e in rep.result.events:
        if e.type is not EventType.DATA_ISSUE:
            by_eid.setdefault(e.identity[0], []).append(e)
    out: list[EpisodeRow] = []
    for eid, evs in by_eid.items():
        start = next(e for e in evs if e.type is EventType.START)

        def first(kind: EventType, evs: list[Event] = evs) -> tuple[date, float | None] | None:
            e = next((x for x in evs if x.type is kind), None)
            return None if e is None else (e.day, closes.get(e.day))

        end_ev = next((e for e in evs if e.type in (EventType.RELEASE, EventType.DATA_BREAK)), None)
        if end_ev is not None:
            end_day: date | None = end_ev.day
            reason: EndReason | None = (
                EndReason.RELEASE if end_ev.type is EventType.RELEASE else EndReason.DATA_BREAK
            )
        elif rep.result.censored_episode == eid:
            end_day, reason = days[-1], results[-1].record.end_reason
        else:
            end_day, reason = None, None
        recs = [r.record for r in results if r.record.episode_id == eid and r.record.l_ep]
        lo, hi = days.index(start.day), days.index(end_day) if end_day else len(days) - 1
        span = results[lo : hi + 1]
        count = {k: sum(1 for e in evs if e.type is k) for k in EventType}
        out.append(
            EpisodeRow(
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
        )
    return _link_previous(out, days, closes)


def _link_previous(
    rows: list[EpisodeRow], days: list[date], closes: Mapping[date, float]
) -> list[EpisodeRow]:
    """第 1A.7 节的关联字段：只是输出字段，不改变状态机。"""
    valid = [d for d in days if d in closes]
    linked: list[EpisodeRow] = []
    prev: EpisodeRow | None = None
    for row in sorted(rows, key=lambda r: r.start_date):
        if prev is None:
            linked.append(row)
        else:
            gap = None
            if prev.end_reason is EndReason.RELEASE and prev.end_date is not None:
                gap = sum(1 for d in valid if prev.end_date < d <= row.start_date)
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
        return ("", "") if p is None else (p[0].isoformat(), _cell(p[1]))

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
                        _cell(e.start_close),
                        json.dumps(list(e.start_reasons)),
                        e.initial_level,
                        _cell(e.low_date),
                        _cell(e.low_close),
                        *st,
                        *ob,
                        *dg,
                        _cell(e.end_date),
                        _cell(e.end_close),
                        _cell(e.end_reason),
                        e.n_recovery_obs,
                        e.n_revoked,
                        e.n_recovery_fail,
                        e.n_escalate,
                        e.duration_trading_days,
                        e.observed_valid_days,
                        _cell(e.entry_path),
                        _cell(e.delta_start),
                        _cell(e.prev_episode_id),
                        _cell(e.days_since_prev_release),
                        _cell(e.start_below_prev_l_ep),
                    ],
                    strict=True,
                )
            )
        )
    return out


def check_consistency(
    events: Sequence[Mapping[str, Any]],
    daily: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    chart_events: Iterable[tuple[str, str]],
) -> list[str]:
    """同一事件在 events、daily、episodes 与图表中的日期一致（读回写出的文件后检查）。

    chart_events：从 chart.html 读回的（日期, 事件类型）。
    """
    problems: list[str] = []
    day_rows = {r["day"]: r for r in daily}
    ev_pairs = sorted((r["date"], r["event_type"]) for r in events)
    if ev_pairs != sorted(chart_events):
        problems.append("图表中的事件标记与 events.csv 不一致")
    expect = {
        "START": lambda d: d["transition"] in ("T6", "T7"),
        "ESCALATE": lambda d: d["transition"] in ("T9", "T10", "T11"),
        "DOWNGRADE": lambda d: d["transition"] == "T15",
        "RELEASE": lambda d: d["end_reason"] == "RELEASE",
        "DATA_BREAK": lambda d: d["end_reason"] == "DATA_BREAK",
        "DATA_ISSUE": lambda d: (
            d["data_status"] in (DataStatus.MISSING.value, DataStatus.DATA_BREAK.value)
        ),
    }
    for r in events:
        d = day_rows.get(r["date"])
        if d is None:
            problems.append(f"{r['date']} {r['event_type']}：daily.csv 中没有这一天")
            continue
        check = expect.get(r["event_type"])
        if check is not None and not check(d):
            problems.append(f"{r['date']} {r['event_type']}：与 daily.csv 当日记录不符")
        if (
            r["episode_id"]
            and r["event_type"] != "DATA_BREAK"
            and d["episode_id"] != r["episode_id"]
        ):
            problems.append(f"{r['date']} {r['event_type']}：daily.csv 的 episode_id 不同")
    level_days = {(r["date"], r["event_type"]) for r in events}
    for d in daily:
        if d["transition"] in ("T6", "T7") and (d["day"], "START") not in level_days:
            problems.append(f"{d['day']}：daily.csv 有启动转移但 events.csv 没有 START")
    for ep in episodes:
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
