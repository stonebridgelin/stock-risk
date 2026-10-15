"""对照回放一套的逐对象输出：events、episodes、紧凑逐日、覆盖逐日、事件对照与重叠表的累积与写出。

从 worker.py 拆出（T08a D-03）。纯数据整理，写出经注入的端口。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from stock_risk.application.compare.views import RunView
from stock_risk.application.replay import SymbolReplay
from stock_risk.application.replay_chart_data import ChartData
from stock_risk.application.replay_episodes import episode_csv_rows, episode_rows
from stock_risk.application.replay_outputs import EPISODE_COLUMNS, EVENT_COLUMNS, event_rows
from stock_risk.domain.lifecycle.runner import RunResult

COMPACT_COLUMNS = ["target", "day", "data_status", "display", "level", "episode_id"]
COVERAGE_COLUMNS = [
    "target",
    "day",
    "v0_display",
    "v0_level",
    "v0_episode_id",
    "new_display",
    "new_level",
    "new_episode_id",
]
OVERLAP_COLUMNS = [
    "target",
    "v0_episode_id",
    "new_episode_id",
    "overlap_days",
    "first_day",
    "last_day",
]
_COMPARE_ORDER = [
    "target",
    "direction",
    "event_type",
    "date",
    "episode_id",
    "close",
    "other_status",
    "other_episode_id",
    "first_20",
    "wait_20",
    "min_close_rel_20",
    "end_close_rel_20",
    "first_60",
    "wait_60",
    "min_close_rel_60",
    "end_close_rel_60",
    "partner_episode_id",
    "partner_end_date",
    "partner_end_reason",
    "release_day_diff",
    "tags",
]


class CompareSink(Protocol):
    def set_dir(self, run_dir: Path, set_id: str) -> Path: ...

    def write_csv(
        self, path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
    ) -> None: ...

    def write_csv_gz(
        self, path: Path, columns: Sequence[str], rows: Sequence[Sequence[Any]]
    ) -> None: ...

    def write_full(
        self,
        run_dir: Path,
        set_id: str,
        target: str,
        daily: Sequence[Mapping[str, Any]],
        chart: ChartData,
    ) -> None: ...


@dataclass(slots=True)
class SetRows:
    """一套在全部对象上累积的明细行，最后一次写出。"""

    set_id: str
    events: list[dict[str, Any]] = field(default_factory=list)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    compact: list[tuple] = field(default_factory=list)
    coverage: list[tuple] = field(default_factory=list)
    compare: list[dict[str, Any]] = field(default_factory=list)
    overlap: list[tuple] = field(default_factory=list)

    def add_target(self, rep: SymbolReplay, result: RunResult, view: RunView, run_id: str) -> None:
        label = rep.target.label
        self.events += [{"set_id": self.set_id, **r} for r in event_rows(rep, run_id)]
        self.episodes += [
            {"set_id": self.set_id, **r} for r in episode_csv_rows(rep, run_id, episode_rows(rep))
        ]
        self.compact += [
            (
                label,
                r.record.day.isoformat(),
                r.record.data_status.value,
                r.record.display.value,
                view.level[i],
                view.episode[i] or "",
            )
            for i, r in enumerate(result.results)
        ]

    def write(self, sink: CompareSink, run_dir: Path, with_compare: bool) -> None:
        d = sink.set_dir(run_dir, self.set_id)
        sink.write_csv(d / "events.csv", ["set_id", *EVENT_COLUMNS], self.events)
        sink.write_csv(d / "episodes.csv", ["set_id", *EPISODE_COLUMNS], self.episodes)
        sink.write_csv_gz(d / "daily_compact.csv.gz", COMPACT_COLUMNS, self.compact)
        if not with_compare:
            return
        sid = self.set_id
        sink.write_csv_gz(d / f"coverage_daily_{sid}.csv.gz", COVERAGE_COLUMNS, self.coverage)
        cols = sorted({k for r in self.compare for k in r}, key=_compare_order)
        sink.write_csv(d / f"event_compare_{sid}.csv", cols, self.compare)
        sink.write_csv(
            d / f"overlap_{sid}.csv",
            OVERLAP_COLUMNS,
            [dict(zip(OVERLAP_COLUMNS, r, strict=True)) for r in self.overlap],
        )


def _compare_order(k: str) -> int:
    return _COMPARE_ORDER.index(k) if k in _COMPARE_ORDER else len(_COMPARE_ORDER)


def coverage_check(
    set_id: str, label: str, base: RunView, new: RunView, rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    """事件对照表覆盖每个 V0 START 与每个新变体 START（统计期间内）。"""
    problems = []
    for view, direction in ((base, "V0→新"), (new, "新→V0")):
        starts = {
            e.day.isoformat()
            for e in view.events
            if e.type.value == "START" and view.in_period(view.index[e.day])
        }
        listed = {
            r["date"] for r in rows if r["direction"] == direction and r["event_type"] == "START"
        }
        if starts != listed:
            problems.append(f"{set_id}/{label}：事件对照表缺少 {direction} 的 START")
    return problems
