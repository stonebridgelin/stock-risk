"""付费档输出的来源主机记录（T08d-2c 第 1 节）。

付费档每个输出文件与 _progress.json 都记录 source_host：CSV 为一列，JSON 为顶层键。
同一输出目录（含子目录）中已有其他主机的记录时，拒绝运行，不混写。
没有 source_host 的旧文件（T08d-2c 之前写出，例如改名保存的 401 进度）不视为任何主机的记录。
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from stock_risk.infrastructure.storage.massive_probe_files import write_csv

SOURCE_HOST = "source_host"


class SourceHostConflict(RuntimeError):
    """输出目录中已有其他主机的记录。"""


def recorded_hosts(root: Path) -> set[str]:
    """root 及其子目录中所有 CSV、JSON 文件记录的 source_host。"""
    hosts: set[str] = set()
    if not root.exists():
        return hosts
    for path in sorted(root.rglob("*")):
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and SOURCE_HOST in reader.fieldnames:
                    hosts.update(r[SOURCE_HOST] for r in reader if r.get(SOURCE_HOST))
        elif path.suffix == ".json":
            try:
                data = json.loads(path.read_text("utf-8"))
            except ValueError:
                continue
            if isinstance(data, dict) and data.get(SOURCE_HOST):
                hosts.add(str(data[SOURCE_HOST]))
    return hosts


def check_source_host(root: Path, host: str) -> None:
    """root 中已有其他主机的记录时抛 SourceHostConflict。"""
    others = sorted(recorded_hosts(root) - {host})
    if others:
        raise SourceHostConflict(
            f"{root} 中已有其他主机（{'、'.join(others)}）的记录，拒绝混写；请换输出目录"
        )


def write_host_csv(
    path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]], host: str
) -> None:
    """写 CSV，并在每行末尾加 source_host 列。"""
    write_csv(path, [*columns, SOURCE_HOST], [{**r, SOURCE_HOST: host} for r in rows])


def write_host_json(path: Path, data: Mapping[str, Any], host: str, indent: int = 2) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps({SOURCE_HOST: host, **data}, ensure_ascii=False, indent=indent)
    tmp.write_text(text + "\n", "utf-8")
    tmp.replace(path)
