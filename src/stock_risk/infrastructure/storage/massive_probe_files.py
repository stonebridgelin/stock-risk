"""Massive 试验的进度与输出（实现 application.probe_massive.ProbeStore），目录 data/massive_probe/：

_progress.json      每个查询组合与案例的状态、封存处理后的行（断点续传依据；不含原始响应）
lists_<日期>.csv     第 2.1 节：该日期各查询组合的代码记录
cases.csv           第 2.2 节：已知案例
summary.json        计数、状态与主交易所变化
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from stock_risk.domain.massive_probe import ADMITTED_FIELDS

PROGRESS_FILE = "_progress.json"
DATE_COLUMNS = ["list_date", "delisted", "delisted_after_2019"]
LIST_COLUMNS = ["query_exchange", "query_type", "query_active", *ADMITTED_FIELDS, *DATE_COLUMNS]
CASE_COLUMNS = [
    "category",
    "ticker",
    "query_date",
    "status",
    "http_status",
    "message",
    *[f for f in ADMITTED_FIELDS if f != "ticker"],
    *DATE_COLUMNS,
]


def _cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return v


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        w.writerows({c: _cell(r.get(c)) for c in columns} for r in rows)
    tmp.replace(path)


class MassiveProbeFiles:
    def __init__(self, out_dir: Path) -> None:
        self._dir = out_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._progress = out_dir / PROGRESS_FILE

    def load_progress(self) -> dict[str, Any]:
        if not self._progress.exists():
            return {}
        return json.loads(self._progress.read_text("utf-8"))

    def save_progress(self, progress: Mapping[str, Any]) -> None:
        tmp = self._progress.with_suffix(".tmp")
        tmp.write_text(json.dumps(progress, ensure_ascii=False, indent=1) + "\n", "utf-8")
        tmp.replace(self._progress)

    def write_outputs(
        self,
        lists: Mapping[date, Sequence[Mapping[str, Any]]],
        cases: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> list[str]:
        paths: list[Path] = []
        for d, rows in sorted(lists.items()):
            p = self._dir / f"lists_{d.isoformat()}.csv"
            write_csv(p, LIST_COLUMNS, rows)
            paths.append(p)
        p = self._dir / "cases.csv"
        write_csv(p, CASE_COLUMNS, cases)
        paths.append(p)
        p = self._dir / "summary.json"
        p.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8")
        paths.append(p)
        return [str(x) for x in paths]
