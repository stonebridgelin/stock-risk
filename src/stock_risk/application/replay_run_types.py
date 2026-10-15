"""回放运行的输出端口、运行报告与离散结果摘要（T04；T08a D-03 从 replay_run 拆出）。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from stock_risk.application.replay_chart_data import ChartData
from stock_risk.domain.lifecycle.runner import RunResult


class ReplayOutputSink(Protocol):
    def create_run_dir(self, run_id: str) -> Path: ...

    def run_dir_exists(self, run_id: str) -> bool: ...

    def write_target(
        self,
        run_dir: Path,
        candidate_id: int,
        label: str,
        events: Sequence[Mapping[str, Any]],
        daily: Sequence[Mapping[str, Any]],
        episodes: Sequence[Mapping[str, Any]],
        chart: ChartData,
    ) -> Path: ...

    def read_target(
        self, run_dir: Path, candidate_id: int, label: str
    ) -> tuple[
        list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[tuple[str, str]]
    ]: ...

    def write_summary(
        self,
        run_dir: Path,
        rows: Sequence[Mapping[str, Any]],
        by_candidate: Sequence[Mapping[str, Any]],
    ) -> list[Path]: ...

    def write_manifest(self, run_dir: Path, manifest: Mapping[str, Any]) -> Path: ...

    def write_index(
        self, run_dir: Path, run_id: str, run_kind: str, cells: Sequence[Mapping[str, Any]]
    ) -> Path: ...


@dataclass(slots=True)
class RunReport:
    run_id: str
    run_kind: str
    run_dir: Path
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        m = self.manifest
        return (
            not m.get("failed")
            and m.get("invariant_violations", 0) == 0
            and not m.get("consistency_problems")
            and m.get("i17_ok", True)
        )


def discrete_digest(result: RunResult) -> tuple:
    """离散结果：每日显示状态、等级、预警编号、周期与当日事件的业务身份（I17 ②）。"""
    return tuple(
        (
            r.record.day,
            r.record.display.value,
            r.record.level,
            r.record.episode_id,
            r.record.cycle_no,
            tuple(sorted(str(e.key) for e in r.events)),
        )
        for r in result.results
    )
