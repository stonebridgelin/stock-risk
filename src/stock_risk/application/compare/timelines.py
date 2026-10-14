"""T06c 单股年度通知时间线：两套并排，取自 T06b 的连续回放（不重新初始化），注明年初状态。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from stock_risk.application.compare.variants import VariantSet
from stock_risk.application.compare.views import RunView
from stock_risk.domain.lifecycle.runner import RunResult

TIMELINE_STOCKS = {
    "KO": (2016,),
    "JNJ": (2016,),
    "AMZN": (2018,),
    "NVDA": (2018,),
    "VRX": (2015, 2016),
    "GE": (2017, 2018),
}


class TimelineSink(Protocol):
    def set_dir(self, run_dir: Path, set_id: str) -> Path: ...

    def write_csv(
        self, path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
    ) -> None: ...

    def read_csv(self, path: Path) -> list[dict[str, str]]: ...

    def write_timeline_index(
        self, run_dir: Path, run_id: str, files: Sequence[Mapping[str, Any]]
    ) -> Path: ...


def timeline_rows(set_id: str, view: RunView, result: RunResult) -> list[dict[str, Any]]:
    """T06c：指定股票与年份的通知（取自连续回放，不重新初始化）及年初状态。"""
    rows: list[dict[str, Any]] = []
    years = TIMELINE_STOCKS[view.target.split("_")[0]]
    recs = [r.record for r in result.results]
    prev_level: dict[date, Any] = {}
    last = None
    for r in recs:
        prev_level[r.day] = last
        last = r.level
    for y in years:
        before = [i for i, d in enumerate(view.days) if d.year < y]
        state = recs[before[-1]] if before else None
        open_ep = state.episode_id if state is not None and state.level is not None else None
        rows.append(
            {
                "set_id": set_id,
                "target": view.target,
                "year": y,
                "kind": "年初状态",
                "date": "",
                "event_type": "",
                "detail": (
                    f"年初已有预警 {open_ep}（启动日 {open_ep.split(':')[1]}，"
                    f"等级 {state.level.name}）"
                    if open_ep and state
                    else "年初无预警"
                ),
            }
        )
        for e in result.events:
            if e.day.year != y:
                continue
            basis = e.basis
            reasons = basis.get("reasons") or ([basis["reason"]] if "reason" in basis else [])
            rows.append(
                {
                    "set_id": set_id,
                    "target": view.target,
                    "year": y,
                    "kind": "通知",
                    "date": e.day.isoformat(),
                    "event_type": e.type.value,
                    "level_before": getattr(prev_level[e.day], "name", ""),
                    "level_after": getattr(recs[view.index[e.day]].level, "name", ""),
                    "reasons": ";".join(map(str, reasons)),
                    "close": view.closes.get(e.day, ""),
                }
            )
    return rows


def build_timelines(
    sink: TimelineSink,
    run_dir: Path,
    run_id: str,
    outcomes: Mapping[str, Any],
    v0: VariantSet,
    v3: VariantSet,
) -> dict[str, Any]:
    """T06c：两套并排的单股年度通知时间线，并与 events.csv 做一致性检查。"""
    rows = [r for sid in (v0.set_id, v3.set_id) if sid in outcomes for r in outcomes[sid].timeline]
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["target"], r["year"])].append(r)
    problems, files = [], []
    events = {
        sid: {
            (e["target"], e["date"], e["event_type"])
            for e in sink.read_csv(sink.set_dir(run_dir, sid) / "events.csv")
        }
        for sid in (v0.set_id, v3.set_id)
        if sid in outcomes
    }
    tl_dir = run_dir / "timelines"
    for (target, year), items in sorted(groups.items()):
        for r in items:
            if r["kind"] == "通知" and (
                r["target"],
                r["date"],
                r["event_type"],
            ) not in events.get(r["set_id"], set()):
                problems.append(
                    f"时间线 {target} {year}：{r['date']} {r['event_type']} "
                    f"不在 {r['set_id']} 的 events.csv 中"
                )
        name = f"{target.split('_')[0]}_{year}.csv"
        cols = [
            "set_id",
            "target",
            "year",
            "kind",
            "date",
            "event_type",
            "level_before",
            "level_after",
            "reasons",
            "close",
            "detail",
        ]
        tl_dir.mkdir(parents=True, exist_ok=True)
        ordered = sorted(items, key=lambda r: (r["kind"] != "年初状态", r["date"], r["set_id"]))
        sink.write_csv(tl_dir / name, cols, ordered)
        files.append(
            {
                "file": name,
                "target": target,
                "year": year,
                "rows": ordered,
                "sets": [v0.set_id, v3.set_id],
            }
        )
    sink.write_timeline_index(run_dir, run_id, files)
    return {"files": [f["file"] for f in files], "problems": problems}
