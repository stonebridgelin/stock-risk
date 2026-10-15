"""把审计结果写成明细 CSV 与 summary.md（实现 application.ports.AuditReportWriter）。"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from datetime import date
from pathlib import Path
from typing import Any

from stock_risk.application.audit_models import AuditReport, SymbolAudit
from stock_risk.infrastructure.storage.audit_summary import fmt_value, render_summary


def _write_csv(path: Path, header: list[str], rows: Iterable[list[Any]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([fmt_value(v) for v in r] for r in rows)
    return path


SYMBOL_COLUMNS = [
    "symbol",
    "rows",
    "first_day",
    "last_day",
    "listed_from",
    "listed_until",
    "rows_outside_listing",
    "gap_check_from",
    "gap_check_to",
    "explicit_failures",
    "adj_factor_jumps",
    "calendar_gaps",
    "zero_volume_days",
    "sealed_rows",
]
JUMP_COLUMNS = ["symbol", "date", "prev_date", "prev_factor", "factor", "change", "in_listing"]
JUNCTION_COLUMNS = [
    "symbol",
    "prev_ticker",
    "next_ticker",
    "prev_date",
    "next_date",
    "prev_close",
    "next_open",
    "open_to_close_ratio",
    "prev_adj_factor",
    "next_adj_factor",
    "adj_factor_change",
]


def _in_listing(s: SymbolAudit, day: date) -> bool:
    return s.listing.contains(day) if s.listing else True


def _symbol_rows(syms: list[SymbolAudit]) -> Iterator[list[Any]]:
    for s in syms:
        yield [
            s.symbol,
            s.rows,
            s.first_day,
            s.last_day,
            s.listing.listed_from if s.listing else None,
            s.listing.listed_until if s.listing else None,
            s.rows_outside_listing,
            s.gap_start,
            s.gap_end,
            len(s.failures),
            len(s.factor_jumps),
            len(s.gaps),
            len(s.zero_volume),
            s.sealed_rows,
        ]


def _jump_rows(syms: list[SymbolAudit], corrected: bool) -> Iterator[list[Any]]:
    for s in syms:
        for j in s.corrected_factor_jumps if corrected else s.factor_jumps:
            yield [
                s.symbol,
                j.day,
                j.prev_day,
                j.prev_factor,
                j.factor,
                j.change,
                _in_listing(s, j.day),
            ]


def _junction_rows(syms: list[SymbolAudit]) -> Iterator[list[Any]]:
    for s in syms:
        for j in s.junctions:
            yield [
                j.symbol,
                j.prev_ticker,
                j.next_ticker,
                j.prev_day,
                j.next_day,
                j.prev_close,
                j.next_open,
                j.open_to_close_ratio,
                j.prev_adj_factor,
                j.next_adj_factor,
                j.adj_factor_change,
            ]


def _tables(report: AuditReport) -> list[tuple[str, list[str], Iterable[list[Any]]]]:
    """（文件名, 表头, 行）。"""
    syms = report.symbols
    return [
        ("symbols.csv", SYMBOL_COLUMNS, _symbol_rows(syms)),
        (
            "explicit_failures.csv",
            ["symbol", "date", "basis", "kind", "detail", "in_listing"],
            (
                [s.symbol, f.day, f.basis or "volume", f.kind, f.detail, _in_listing(s, f.day)]
                for s in syms
                for f in s.failures
            ),
        ),
        ("adj_factor_jumps.csv", JUMP_COLUMNS, _jump_rows(syms, corrected=False)),
        ("calendar_gaps.csv", ["symbol", "date"], ([s.symbol, d] for s in syms for d in s.gaps)),
        (
            "zero_volume.csv",
            ["symbol", "date", "in_listing"],
            ([s.symbol, d, _in_listing(s, d)] for s in syms for d in s.zero_volume),
        ),
        ("junctions.csv", JUNCTION_COLUMNS, _junction_rows(syms)),
        (
            "price_corrections.csv",
            ["symbol", "from", "to", "method", "ruling", "evidence"],
            (
                [c.symbol, c.start, c.end, c.method, c.ruling, c.evidence]
                for s in syms
                for c in s.corrections
            ),
        ),
        ("corrected_adj_factor_jumps.csv", JUMP_COLUMNS, _jump_rows(syms, corrected=True)),
        (
            "manifest_check.csv",
            ["file", "status", "expected_sha256", "actual_sha256"],
            ([m.file, m.status, m.expected, m.actual] for m in report.manifest_mismatches),
        ),
    ]


class AuditReportFileWriter:
    def write(self, report: AuditReport, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = [_write_csv(out_dir / name, head, rows) for name, head, rows in _tables(report)]
        summary = out_dir / "summary.md"
        summary.write_text(render_summary(report), "utf-8")
        paths.append(summary)
        return paths
