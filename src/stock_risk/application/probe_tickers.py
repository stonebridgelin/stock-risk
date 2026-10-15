"""用例：代码清单探查 `probe-tickers`（T08d-1 第 1 节）。

只读取数据源公开的代码清单（一个请求），不下载任何价格。清单在内存中逐行做封存处理
（domain/universe_probe.py），原始压缩包不落盘；只写出封存处理后的结果。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from stock_risk.domain.universe_probe import (
    CoverageRequest,
    TickerRow,
    coverage,
    is_selected,
    seal_rows,
    summarize,
)


class TickerListSource(Protocol):
    def rows(self) -> Iterator[Mapping[str, str]]:
        """逐行给出清单的原始字段（只在内存中）；字段不符时抛 ProbeFormatError。"""
        ...


class ProbeWriter(Protocol):
    def write(
        self,
        out_dir: Path,
        rows: Sequence[TickerRow],
        summary: Mapping[str, Any],
        checked: Sequence[Mapping[str, Any]] | None,
    ) -> list[Path]: ...


@dataclass(frozen=True, slots=True)
class ProbeResult:
    summary: dict[str, Any]
    checked: list[dict[str, Any]] | None
    paths: list[Path]


class ProbeTickersService:
    def __init__(self, source: TickerListSource, writer: ProbeWriter) -> None:
        self._source = source
        self._writer = writer

    def run(self, out_dir: Path, check: Sequence[CoverageRequest] | None = None) -> ProbeResult:
        sealed = seal_rows(self._source.rows())
        summary = summarize(sealed)
        checked = coverage(sealed, check) if check is not None else None
        selected = sorted(
            (r for r in sealed.rows if is_selected(r)), key=lambda r: (r.ticker, r.start)
        )
        paths = self._writer.write(out_dir, selected, summary, checked)
        return ProbeResult(summary, checked, paths)
