"""中转服务核验的进度与输出（实现 application.verify_massive_relay.VerifyStore），
目录 data/massive_probe/paid/verify/（不入库）：

_progress.json                  各步状态与计数（断点续传依据；不含原始响应）
coverage.csv                    第 2 节：接口覆盖试探的结果表
prices_<代码>.csv                第 3.1 节：逐日对账（中转与 Tiingo 两边任一边有的日子）
verify_prices_summary.csv       第 3.1 节：每只股票一行的汇总表
lists/relay_<查询>.csv           第 3.2 节：中转服务的清单（封存后，白名单字段）
verify_reference_summary.csv    第 3.2 节：每个快照日期一行的汇总表
verify_reference_examples.csv   第 3.2 节：只在一边出现与字段不同的例子
summary.json                    请求次数、耗时、封存计数

每个文件都记录 source_host；付费档目录（含本目录）中已有其他主机的记录时，构造时拒绝。
另提供 T08d-2a 官方快照（data/massive_probe/lists_<日期>.csv）的读取。
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from stock_risk.domain.massive_probe import ADMITTED_FIELDS, ListQuery
from stock_risk.domain.massive_verify import REF_FIELDS
from stock_risk.infrastructure.storage.massive_source_host import (
    check_source_host,
    write_host_csv,
    write_host_json,
)

PROGRESS_FILE = "_progress.json"
COVERAGE_COLUMNS = [
    "name",
    "request",
    "status",
    "http_status",
    "pages",
    "rows",
    "dropped_after_2019",
    "result_fields",
    "message",
]
PRICE_COLUMNS = [
    "date",
    "side",
    "in_spy_calendar",
    "relay_close",
    "tiingo_close",
    "close_diff",
    "relay_volume",
    "tiingo_volume",
    "volume_rel_diff",
]
PRICE_SUMMARY_COLUMNS = [
    "ticker",
    "status",
    "relay_rows",
    "tiingo_rows",
    "common_days",
    "relay_only_days",
    "relay_only_dates",
    "tiingo_only_days",
    "tiingo_only_dates",
    "relay_close_missing",
    "max_abs_close_diff",
    "days_over_0_01",
    "share_over_0_01",
    "top5_abs_diff",
    "median_close_diff",
    "volume_rel_diff_median",
    "volume_abs_rel_diff_p95",
    "volume_days_skipped",
    "dropped_after_2019",
    "message",
]
LIST_COLUMNS = [*ADMITTED_FIELDS, "list_date", "delisted", "delisted_after_2019"]
REF_SUMMARY_COLUMNS = [
    "date",
    "status",
    "official_rows",
    "relay_rows",
    "only_official",
    "only_relay",
    "only_either_share",
    "rows_with_field_diff",
    "field_diff_share",
    *[f"diff_{f}" for f in REF_FIELDS],
    "message",
]
REF_EXAMPLE_COLUMNS = [
    "date",
    "kind",
    "field",
    "query",
    "ticker",
    "official_value",
    "relay_value",
]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class MassiveVerifyFiles:
    def __init__(self, paid_dir: Path, source_host: str) -> None:
        check_source_host(paid_dir, source_host)  # 付费档目录整体不混写
        self._dir = paid_dir / "verify"
        self._host = source_host
        (self._dir / "lists").mkdir(parents=True, exist_ok=True)

    @property
    def out_dir(self) -> Path:
        return self._dir

    def load_progress(self) -> dict[str, Any]:
        p = self._dir / PROGRESS_FILE
        return json.loads(p.read_text("utf-8")) if p.exists() else {}

    def save_progress(self, progress: Mapping[str, Any]) -> None:
        write_host_json(self._dir / PROGRESS_FILE, progress, self._host, indent=1)

    def save_price_rows(self, ticker: str, rows: Sequence[Mapping[str, Any]]) -> None:
        write_host_csv(self._dir / f"prices_{ticker}.csv", PRICE_COLUMNS, rows, self._host)

    def save_relay_list(self, key: str, rows: Sequence[Mapping[str, Any]]) -> None:
        write_host_csv(self._dir / "lists" / f"relay_{key}.csv", LIST_COLUMNS, rows, self._host)

    def load_relay_list(self, key: str) -> list[dict[str, str]]:
        return _read(self._dir / "lists" / f"relay_{key}.csv")

    def write_outputs(
        self,
        coverage: Sequence[Mapping[str, Any]],
        prices: Sequence[Mapping[str, Any]],
        reference: Sequence[Mapping[str, Any]],
        examples: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> list[str]:
        outputs = (
            ("coverage.csv", COVERAGE_COLUMNS, coverage),
            ("verify_prices_summary.csv", PRICE_SUMMARY_COLUMNS, prices),
            ("verify_reference_summary.csv", REF_SUMMARY_COLUMNS, reference),
            ("verify_reference_examples.csv", REF_EXAMPLE_COLUMNS, examples),
        )
        paths: list[Path] = []
        for name, columns, rows in outputs:
            p = self._dir / name
            write_host_csv(p, columns, rows, self._host)
            paths.append(p)
        p = self._dir / "summary.json"
        write_host_json(p, summary, self._host)
        paths.append(p)
        return [str(x) for x in paths]


def read_official_snapshots(
    probe_dir: Path, queries: Sequence[ListQuery]
) -> dict[str, list[dict[str, str]]]:
    """T08d-2a 官方免费档快照：按查询键分组的行（只读，不发请求）。快照文件不存在的日期不列出。"""
    out: dict[str, list[dict[str, str]]] = {}
    files: dict[date, list[dict[str, str]] | None] = {}
    for q in queries:
        if q.day not in files:
            p = probe_dir / f"lists_{q.day.isoformat()}.csv"
            files[q.day] = _read(p) if p.exists() else None
        rows = files[q.day]
        if rows is None:
            continue
        flag = "true" if q.active else "false"
        out[q.key] = [
            r
            for r in rows
            if r["query_exchange"] == q.exchange
            and r["query_type"] == q.type
            and r["query_active"] == flag
        ]
    return out
