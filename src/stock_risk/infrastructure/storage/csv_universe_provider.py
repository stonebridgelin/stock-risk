"""从 CSV 读取研究样本（实现 application.ports.UniverseProvider）。

格式：symbol,source_candidates,category,note
source_candidates 用 | 分隔，按顺序尝试。以 # 开头的行为注释。
"""

from __future__ import annotations

import csv
from pathlib import Path

from stock_risk.domain.universe import UniverseMember


class CsvUniverseProvider:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> list[UniverseMember]:
        with self._path.open(encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
        members = [
            UniverseMember(
                symbol=r["symbol"].strip().upper(),
                source_candidates=tuple(
                    c.strip().upper() for c in r["source_candidates"].split("|") if c.strip()
                ),
                category=r["category"].strip(),
                note=(r.get("note") or "").strip(),
            )
            for r in csv.DictReader(lines)
        ]
        symbols = [m.symbol for m in members]
        dup = {s for s in symbols if symbols.count(s) > 1}
        if dup:
            raise ValueError(f"研究样本中代码重复: {sorted(dup)}")
        return members
