"""把审计结果写成明细 CSV 与 summary.md（实现 application.ports.AuditReportWriter）。"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from stock_risk.application.audit_models import AuditReport, SymbolAudit


def _fmt(v: Any, digits: int = 6) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _write_csv(path: Path, header: list[str], rows: Iterable[list[Any]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([_fmt(v) for v in r] for r in rows)
    return path


def _listing_text(s: SymbolAudit) -> str:
    if s.listing is None or not s.listing.is_bounded:
        return "" if s.listing is None else "已登记，日期待核实"
    return f"{_fmt(s.listing.listed_from) or '…'} ~ {_fmt(s.listing.listed_until) or '…'}"


class AuditReportFileWriter:
    def write(self, report: AuditReport, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        syms = report.symbols
        paths = [
            _write_csv(
                out_dir / "symbols.csv",
                [
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
                ],
                (
                    [
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
                    for s in syms
                ),
            ),
            _write_csv(
                out_dir / "explicit_failures.csv",
                ["symbol", "date", "basis", "kind", "detail", "in_listing"],
                (
                    [
                        s.symbol,
                        f.day,
                        f.basis or "volume",
                        f.kind,
                        f.detail,
                        s.listing.contains(f.day) if s.listing else True,
                    ]
                    for s in syms
                    for f in s.failures
                ),
            ),
            _write_csv(
                out_dir / "adj_factor_jumps.csv",
                ["symbol", "date", "prev_date", "prev_factor", "factor", "change", "in_listing"],
                (
                    [
                        s.symbol,
                        j.day,
                        j.prev_day,
                        j.prev_factor,
                        j.factor,
                        j.change,
                        s.listing.contains(j.day) if s.listing else True,
                    ]
                    for s in syms
                    for j in s.factor_jumps
                ),
            ),
            _write_csv(
                out_dir / "calendar_gaps.csv",
                ["symbol", "date"],
                ([s.symbol, d] for s in syms for d in s.gaps),
            ),
            _write_csv(
                out_dir / "zero_volume.csv",
                ["symbol", "date", "in_listing"],
                (
                    [s.symbol, d, s.listing.contains(d) if s.listing else True]
                    for s in syms
                    for d in s.zero_volume
                ),
            ),
            _write_csv(
                out_dir / "junctions.csv",
                [
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
                ],
                (
                    [
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
                    for s in syms
                    for j in s.junctions
                ),
            ),
        ]
        summary = out_dir / "summary.md"
        summary.write_text(_summary(report), "utf-8")
        paths.append(summary)
        return paths


def _summary(r: AuditReport) -> str:
    lines = [
        f"# 原始数据审计 {r.run_date}",
        "",
        f"- 数据版本（MANIFEST.sha256 文件的 SHA-256）：`{r.manifest_sha256 or '无清单'}`",
        f"- 交易日历代理：{r.calendar_symbol}（{r.calendar_first} ~ {r.calendar_last}）",
        f"- 复权系数跳变阈值：{r.adj_factor_threshold:.0%}（任务卡 T00 给定）",
        f"- 审计代码 {len(r.symbols)} 个；研究样本中缺少原始文件 {len(r.missing)} 个"
        + (f"：{', '.join(r.missing)}" if r.missing else ""),
        f"- 明确失败合计 {r.total_failures} 项；封存越界行合计 {r.total_sealed_rows} 行（应为 0）",
        "",
        "本审计只报告，不修改任何数据。明细见同目录 CSV。",
        "",
        "## 各代码概览",
        "",
        "| 代码 | 行数 | 首日 | 末日 | 上市期间 | 上市期外行数 | 明确失败 | 系数跳变 "
        "| 缺口 | 零成交量 | 封存越界 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in r.symbols:
        lines.append(
            f"| {s.symbol} | {s.rows} | {s.first_day} | {s.last_day} | {_listing_text(s)} | "
            f"{s.rows_outside_listing} | {len(s.failures)} | {len(s.factor_jumps)} | "
            f"{len(s.gaps)} | {len(s.zero_volume)} | {s.sealed_rows} |"
        )

    lines += ["", "## 明确失败（v0.4 §2.0）", ""]
    fails = [(s, f) for s in r.symbols for f in s.failures]
    if not fails:
        lines.append("无。")
    else:
        lines += ["| 代码 | 日期 | 口径 | 类型 | 明细 |", "| --- | --- | --- | --- | --- |"]
        lines += [
            f"| {s.symbol} | {f.day} | {f.basis or 'volume'} | {f.kind} | {f.detail} |"
            for s, f in fails
        ]

    lines += ["", "## 复权系数跳变（当天无公司行动记录）", ""]
    jumps = [(s, j) for s in r.symbols for j in s.factor_jumps]
    if not jumps:
        lines.append("无。")
    else:
        lines += ["| 代码 | 日期 | 前一日 | 系数变化 |", "| --- | --- | --- | --- |"]
        lines += [f"| {s.symbol} | {j.day} | {j.prev_day} | {j.change:+.2%} |" for s, j in jumps]

    lines += ["", "## 交易日缺口", ""]
    gapped = [s for s in r.symbols if s.gaps]
    if not gapped:
        lines.append("无。")
    else:
        lines += ["| 代码 | 检查区间 | 缺口天数 | 日期 |", "| --- | --- | --- | --- |"]
        lines += [
            f"| {s.symbol} | {s.gap_start} ~ {s.gap_end} | {len(s.gaps)} | "
            f"{', '.join(str(d) for d in s.gaps)} |"
            for s in gapped
        ]

    lines += ["", "## 零成交量", ""]
    zv = [s for s in r.symbols if s.zero_volume]
    if not zv:
        lines.append("无。")
    else:
        lines += [f"- {s.symbol}：{', '.join(str(d) for d in s.zero_volume)}" for s in zv]

    lines += ["", "## 拼接点", ""]
    js = [j for s in r.symbols for j in s.junctions]
    if not js:
        lines.append("无。")
    else:
        lines += [
            "| 代码 | 前段 → 后段 | 前段末日 | 后段首日 | 前收 | 后开 | 后开/前收 "
            "| 系数前 | 系数后 | 系数变化 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {j.symbol} | {j.prev_ticker} → {j.next_ticker} | {j.prev_day} | {j.next_day} | "
            f"{_fmt(j.prev_close, 4)} | {_fmt(j.next_open, 4)} | "
            f"{_fmt(j.open_to_close_ratio, 4)} | "
            f"{_fmt(j.prev_adj_factor)} | {_fmt(j.next_adj_factor)} | "
            + (f"{j.adj_factor_change:+.4%}" if j.adj_factor_change is not None else "")
            + " |"
            for j in js
        ]
    return "\n".join(lines) + "\n"
