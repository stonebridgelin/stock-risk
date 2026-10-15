"""Massive 追加探查的进度与输出（实现 application.probe_massive_followup.FollowupStore），
目录 data/massive_probe/followup/：

_progress.json        A 的封存处理后的行、B 的计数（不含 B 的逐行内容）、C 的结果（断点续传依据）
lists_<日期>.csv       A：不传 type 的清单行
crosstab_<日期>.csv    B：primary_exchange × type 计数（空值为"(空)"）
cases.csv             C：补充案例
coverage.csv          D：与 Tiingo 探查按代码对照的逐代码结果
summary.json          A、B 的计数，C 的状态，D 的条数与例子，累计请求数与耗时

另提供 T08d-1 Tiingo 探查输出（tickers_2008_2019.csv）的读取。
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from stock_risk.domain.massive_followup import ReferenceSpan
from stock_risk.domain.massive_probe import ADMITTED_FIELDS
from stock_risk.infrastructure.storage.massive_probe_files import (
    CASE_COLUMNS,
    DATE_COLUMNS,
    MassiveProbeFiles,
    write_csv,
)

LIST_COLUMNS = ["query_exchange", "query_active", *ADMITTED_FIELDS, *DATE_COLUMNS]
CROSSTAB_COLUMNS = ["primary_exchange", "type", "count"]
COVERAGE_COLUMNS = [
    "date",
    "side",
    "ticker",
    "massive_exchange",
    "massive_type",
    "tiingo_exchange",
]


class MassiveFollowupFiles:
    def __init__(self, out_dir: Path) -> None:
        self._dir = out_dir
        self._progress = MassiveProbeFiles(out_dir)  # 进度文件的读写与第一次探查相同

    def load_progress(self) -> dict[str, Any]:
        return self._progress.load_progress()

    def save_progress(self, progress: Mapping[str, Any]) -> None:
        self._progress.save_progress(progress)

    def write_outputs(
        self,
        lists: Mapping[date, Sequence[Mapping[str, Any]]],
        crosstabs: Mapping[date, Mapping[str, Mapping[str, int]]],
        cases: Sequence[Mapping[str, Any]],
        coverage: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> list[str]:
        paths: list[Path] = []
        for d, rows in sorted(lists.items()):
            p = self._dir / f"lists_{d.isoformat()}.csv"
            write_csv(p, LIST_COLUMNS, rows)
            paths.append(p)
        for d, table in sorted(crosstabs.items()):
            p = self._dir / f"crosstab_{d.isoformat()}.csv"
            flat = [
                {"primary_exchange": x, "type": t, "count": n}
                for x, ts in table.items()
                for t, n in ts.items()
            ]
            write_csv(p, CROSSTAB_COLUMNS, flat)
            paths.append(p)
        for name, columns, rows in (
            ("cases.csv", CASE_COLUMNS, cases),
            ("coverage.csv", COVERAGE_COLUMNS, coverage),
        ):
            p = self._dir / name
            write_csv(p, columns, rows)
            paths.append(p)
        p = self._dir / "summary.json"
        p.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8")
        paths.append(p)
        return [str(x) for x in paths]


def read_reference_spans(path: Path) -> list[ReferenceSpan]:
    """读取 T08d-1 的 tickers_2008_2019.csv。

    列：ticker,exchange,start_date,end_date,active_after_2019。
    """
    with path.open(newline="", encoding="utf-8") as f:
        return [
            ReferenceSpan(
                ticker=r["ticker"],
                exchange=r["exchange"],
                start=date.fromisoformat(r["start_date"]),
                end=date.fromisoformat(r["end_date"]),
                active_after_2019=r["active_after_2019"] == "true",
            )
            for r in csv.DictReader(f)
        ]
