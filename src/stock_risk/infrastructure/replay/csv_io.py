"""回放与对照回放共用的 CSV、运行清单读写（T08a D-03）。

数值格式：浮点数写成 10 位有效数字（`{:.10g}`），其余原样。V0 回归按同一规则格式化后比较。
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def fmt(v: Any) -> Any:
    if isinstance(v, float):
        return f"{v:.10g}"
    return v


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        w.writerows({k: fmt(r.get(k, "")) for k in columns} for r in rows)


def write_csv_gz(path: Path, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=6) as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows([fmt(v) for v in r] for r in rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def format_rows(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """按写 CSV 时的同一规则格式化后、经过 CSV 往返的结果（V0 回归比较用）。"""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(columns), extrasaction="ignore")
    w.writeheader()
    w.writerows({k: fmt(r.get(k, "")) for k in columns} for r in rows)
    buf.seek(0)
    return list(csv.DictReader(buf))


def write_manifest(run_dir: Path, manifest: Mapping[str, Any]) -> Path:
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", "utf-8")
    return path
