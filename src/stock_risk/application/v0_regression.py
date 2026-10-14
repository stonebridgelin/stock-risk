"""V0 真实数据回归（T06a 验收）：新代码在全部开关关闭（V0）下重跑 12 组参考候选 × 28 个回放对象，
与一次正式运行的输出逐行逐值比较。只比较基准文件中**原有的全部列**，新增列不参与比较。

比较在内存中进行：新结果按写 CSV 时的同一格式化规则处理后，与基准 CSV 的读回值比较字符串。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from stock_risk.application.replay import ReplayService, SymbolReplay, replay_prepared
from stock_risk.application.replay_outputs import (
    DAILY_FIELDS,
    EPISODE_COLUMNS,
    EVENT_COLUMNS,
    daily_rows,
    episode_csv_rows,
    episode_rows,
    event_rows,
)
from stock_risk.domain.lifecycle.params import (
    EXPLORATION_ASSUMPTIONS,
    V0_SWITCHES,
    EngineParams,
    reference_candidates,
)

TABLE_COLUMNS = {
    "events": EVENT_COLUMNS,
    "daily": ["run_id", "candidate_id", "target", *DAILY_FIELDS, "close"],
    "episodes": EPISODE_COLUMNS,
}


class TableSink(Protocol):
    def format_rows(
        self, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, str]]: ...

    def read_tables(
        self, run_dir: Path, candidate_id: int, label: str
    ) -> dict[str, list[dict[str, str]]]: ...


@dataclass(slots=True)
class RegressionReport:
    baseline_run_id: str
    compared_targets: int = 0
    compared_rows: int = 0
    compared_cells: int = 0
    differences: list[str] = field(default_factory=list)
    difference_count: int = 0

    @property
    def identical(self) -> bool:
        return self.difference_count == 0


def compare_tables(
    base: Sequence[Mapping[str, str]],
    new: Sequence[Mapping[str, str]],
    where: str,
    report: RegressionReport,
) -> None:
    """基准的每一列（原有列）都必须逐行逐值相同；新增列忽略。"""
    if len(base) != len(new):
        report.difference_count += 1
        report.differences.append(f"{where}：行数 {len(base)} → {len(new)}")
        return
    for i, (b, n) in enumerate(zip(base, new, strict=True)):
        report.compared_rows += 1
        for col, value in b.items():
            report.compared_cells += 1
            if n.get(col) != value:
                report.difference_count += 1
                if len(report.differences) < 50:
                    report.differences.append(
                        f"{where} 第 {i + 1} 行 {col}：{value!r} → {n.get(col)!r}"
                    )


def run_v0_regression(
    replay: ReplayService,
    sink: TableSink,
    baseline_dir: Path,
    progress: Callable[[str], None] = print,
) -> RegressionReport:
    report = RegressionReport(baseline_dir.name)
    run_id = baseline_dir.name  # 输出中的 run_id 列按基准填写，以便逐值比较
    g = EXPLORATION_ASSUMPTIONS.gap_break
    prepared = [replay.prepare(t, g) for t in replay.targets()]
    calendar = replay.loader.calendar()
    for cid, cand in reference_candidates():
        params = EngineParams(cand, EXPLORATION_ASSUMPTIONS, switches=V0_SWITCHES)
        for p in prepared:
            rep = SymbolReplay(p.target, cid, params, p, replay_prepared(p, params, calendar))
            new = {
                "events": event_rows(rep, run_id),
                "daily": daily_rows(rep, run_id),
                "episodes": episode_csv_rows(rep, run_id, episode_rows(rep)),
            }
            base = sink.read_tables(baseline_dir, cid, p.target.label)
            for name, rows in new.items():
                fmt = sink.format_rows(TABLE_COLUMNS[name], rows)
                compare_tables(base[name], fmt, f"c{cid:03d}/{p.target.label}/{name}", report)
            report.compared_targets += 1
        progress(f"  候选 {cid} 比较完成，累计差异 {report.difference_count}")
    return report
