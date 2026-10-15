"""回放结果 → 输出行（events、daily）与共用的列定义、单元格格式（T04b）。纯函数。

episodes 行见 replay_episodes，四种输出之间的一致性检查见 replay_consistency（T08a D-03 拆分）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from datetime import date
from typing import Any

from stock_risk.application.replay import SymbolReplay
from stock_risk.domain.lifecycle.model import (
    DailyRecord,
    DataStatus,
    EventType,
    Level,
)

DAILY_FIELDS = [f.name for f in fields(DailyRecord)]
# 缺价日的说明（T08a B-04）：display 列保持冻结的显示状态（V0 回归逐值比较），另加 data_note 列
MISSING_NOTE = "当日缺价，状态未更新"
DAILY_COLUMNS = ["run_id", "candidate_id", "target", *DAILY_FIELDS, "close", "data_note"]
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
    "xc_segment",  # T08a B-06：DIRECT 解除时 xc 计数段内每日的证据
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


def jsonable(v: Any) -> Any:
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, (frozenset, set, tuple, list)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, Level):
        return v.name
    return v


def cell(v: Any) -> Any:
    """CSV 单元格：集合与元组写成 JSON，枚举写名称。"""
    if v is None:
        return ""
    if isinstance(v, Level):
        return v.name
    if isinstance(v, (frozenset, set)):
        return json.dumps(sorted(v), ensure_ascii=False)
    if isinstance(v, (tuple, list)):
        return json.dumps(jsonable(list(v)), ensure_ascii=False)
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
        row.update({k: cell(rec[k]) for k in DAILY_FIELDS})
        row["close"] = cell(closes.get(r.record.day))
        row["data_note"] = MISSING_NOTE if r.record.data_status is DataStatus.MISSING else ""
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
        xc_segment = basis.pop("xc_segment", None)
        v08 = basis.pop("v08", None)
        before, after = levels[e.day]
        out.append(
            {
                "run_id": run_id,
                "candidate_id": rep.candidate_id,
                "target": rep.target.label,
                "date": e.day.isoformat(),
                "event_type": e.type.value,
                "identity": json.dumps(jsonable(list(e.identity)), ensure_ascii=False),
                "episode_id": "" if e.type is EventType.DATA_ISSUE else e.identity[0],
                "level_before": cell(before),
                "level_after": cell(after),
                "transition": e.transition or "",
                "reasons": json.dumps(reasons, ensure_ascii=False),
                "evidence": json.dumps(
                    today if today is not None else sorted(recs[e.day].evidence), ensure_ascii=False
                ),
                "segment": "" if segment is None else json.dumps(segment, ensure_ascii=False),
                "values": json.dumps(jsonable(basis), ensure_ascii=False),
                "values_v08": "" if v08 is None else json.dumps(jsonable(v08), ensure_ascii=False),
                "xc_segment": ""
                if xc_segment is None
                else json.dumps(xc_segment, ensure_ascii=False),
            }
        )
    return out
