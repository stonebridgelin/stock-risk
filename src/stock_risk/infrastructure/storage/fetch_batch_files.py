"""fetch-batch 的输入清单与额度状态文件（T08d-1）。

- 下载清单 CSV：symbol,source_ticker,from,to（ISO 日期；to 不得晚于 2019-12-31）。
- `<out>/_quota_state.json`：按 UTC 日期、月份记账的额度计数。
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

from stock_risk.application.fetch_batch import BatchItem

LIST_COLUMNS = ("symbol", "source_ticker", "from", "to")
QUOTA_FILE = "_quota_state.json"


def read_batch_list(path: Path) -> list[BatchItem]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in LIST_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: 缺少列 {missing}")
        return [
            BatchItem(
                r["symbol"].strip().upper(),
                r["source_ticker"].strip().upper(),
                date.fromisoformat(r["from"].strip()),
                date.fromisoformat(r["to"].strip()),
            )
            for r in reader
            if r["symbol"].strip()
        ]


class JsonQuotaStore:
    def __init__(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._path = out_dir / QUOTA_FILE

    def load(self) -> dict[str, Any] | None:
        return json.loads(self._path.read_text("utf-8")) if self._path.exists() else None

    def save(self, state: dict[str, Any]) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", "utf-8")
        tmp.replace(self._path)
