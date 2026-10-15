"""审计 summary.md 的文本生成。"""

from __future__ import annotations

from typing import Any

from stock_risk.application.audit_models import AuditReport, SymbolAudit


def fmt_value(v: Any, digits: int = 6) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _listing_text(s: SymbolAudit) -> str:
    if s.listing is None or not s.listing.is_bounded:
        return "" if s.listing is None else "已登记，日期待核实"
    return f"{fmt_value(s.listing.listed_from) or '…'} ~ {fmt_value(s.listing.listed_until) or '…'}"


def render_summary(r: AuditReport) -> str:
    lines = _header(r)
    for section in (
        _failures_section,
        _jumps_section,
        _gaps_section,
        _zero_volume_section,
        _junctions_section,
        _manifest_section,
        _corrections_section,
    ):
        lines += section(r)
    return "\n".join(lines) + "\n"


def _header(r: AuditReport) -> list[str]:
    lines = [
        f"# 原始数据审计 {r.run_date}",
        "",
        f"- 数据版本（MANIFEST.sha256 文件的 SHA-256）：`{r.manifest_sha256 or '无清单'}`",
        f"- 交易日历代理：{r.calendar_symbol}（{r.calendar_first} ~ {r.calendar_last}）",
        f"- 复权系数跳变阈值：{r.adj_factor_threshold:.0%}（任务卡 T00 给定）",
        f"- 审计代码 {len(r.symbols)} 个；研究样本中缺少原始文件 {len(r.missing)} 个"
        + (f"：{', '.join(r.missing)}" if r.missing else "")
        + (
            f"；下载状态已确认无数据 {len(r.no_data)} 个：{', '.join(r.no_data)}"
            if r.no_data
            else ""
        ),
        f"- 明确失败合计 {r.total_failures} 项；封存越界行合计 {r.total_sealed_rows} 行（应为 0）",
        f"- MANIFEST 校验：清单 {r.manifest_entries} 条，不一致 {len(r.manifest_mismatches)} 项",
        f"- 已登记的复权价更正 {sum(len(s.corrections) for s in r.symbols)} 条（I11）",
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
    return lines


def _failures_section(r: AuditReport) -> list[str]:
    lines = ["", "## 明确失败（v0.4 §2.0）", ""]
    fails = [(s, f) for s in r.symbols for f in s.failures]
    if not fails:
        return [*lines, "无。"]
    lines += ["| 代码 | 日期 | 口径 | 类型 | 明细 |", "| --- | --- | --- | --- | --- |"]
    return lines + [
        f"| {s.symbol} | {f.day} | {f.basis or 'volume'} | {f.kind} | {f.detail} |"
        for s, f in fails
    ]


def _jumps_section(r: AuditReport) -> list[str]:
    lines = ["", "## 复权系数跳变（当天无公司行动记录）", ""]
    jumps = [(s, j) for s in r.symbols for j in s.factor_jumps]
    if not jumps:
        return [*lines, "无。"]
    lines += ["| 代码 | 日期 | 前一日 | 系数变化 |", "| --- | --- | --- | --- |"]
    return lines + [f"| {s.symbol} | {j.day} | {j.prev_day} | {j.change:+.2%} |" for s, j in jumps]


def _gaps_section(r: AuditReport) -> list[str]:
    lines = ["", "## 交易日缺口", ""]
    gapped = [s for s in r.symbols if s.gaps]
    if not gapped:
        return [*lines, "无。"]
    lines += ["| 代码 | 检查区间 | 缺口天数 | 日期 |", "| --- | --- | --- | --- |"]
    return lines + [
        f"| {s.symbol} | {s.gap_start} ~ {s.gap_end} | {len(s.gaps)} | "
        f"{', '.join(str(d) for d in s.gaps)} |"
        for s in gapped
    ]


def _zero_volume_section(r: AuditReport) -> list[str]:
    lines = ["", "## 零成交量", ""]
    zv = [s for s in r.symbols if s.zero_volume]
    if not zv:
        return [*lines, "无。"]
    return lines + [f"- {s.symbol}：{', '.join(str(d) for d in s.zero_volume)}" for s in zv]


def _junctions_section(r: AuditReport) -> list[str]:
    lines = ["", "## 拼接点", ""]
    js = [j for s in r.symbols for j in s.junctions]
    if not js:
        return [*lines, "无。"]
    lines += [
        "| 代码 | 前段 → 后段 | 前段末日 | 后段首日 | 前收 | 后开 | 后开/前收 "
        "| 系数前 | 系数后 | 系数变化 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    return lines + [
        f"| {j.symbol} | {j.prev_ticker} → {j.next_ticker} | {j.prev_day} | {j.next_day} | "
        f"{fmt_value(j.prev_close, 4)} | {fmt_value(j.next_open, 4)} | "
        f"{fmt_value(j.open_to_close_ratio, 4)} | "
        f"{fmt_value(j.prev_adj_factor)} | {fmt_value(j.next_adj_factor)} | "
        + (f"{j.adj_factor_change:+.4%}" if j.adj_factor_change is not None else "")
        + " |"
        for j in js
    ]


def _manifest_section(r: AuditReport) -> list[str]:
    lines = ["", "## MANIFEST 校验", ""]
    if not r.manifest_entries:
        return [*lines, "没有 MANIFEST.sha256，未校验。"]
    if not r.manifest_mismatches:
        return [*lines, f"清单 {r.manifest_entries} 条，全部日线文件哈希一致。"]
    lines += ["| 文件 | 状态 | 清单哈希 | 实际哈希 |", "| --- | --- | --- | --- |"]
    lines += [
        f"| {m.file} | {m.status} | {m.expected or ''} | {m.actual or ''} |"
        for m in r.manifest_mismatches
    ]
    return lines


def _corrections_section(r: AuditReport) -> list[str]:
    lines = [
        "",
        "## 已登记的更正",
        "",
        "以上各节都针对未更正的原始数据。下表为 `config/price_corrections.csv` 中登记的复权价"
        "更正，以及对更正后数据重新做的复权系数跳变检查。",
        "",
    ]
    corrected = [s for s in r.symbols if s.corrections]
    if not corrected:
        return [*lines, "无。"]
    lines += [
        "| 代码 | 区间 | 方法 | 裁定 | 数据依据 |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {c.symbol} | {c.start} ~ {c.end} | {c.method} | {c.ruling} | {c.evidence} |"
        for s in corrected
        for c in s.corrections
    ]
    lines += ["", "更正后的复权系数跳变（当天无公司行动）：", ""]
    for s in corrected:
        if s.corrected_factor_jumps:
            jumps = "，".join(
                f"{j.day} {j.change:+.2%}"
                + ("" if s.listing is None or s.listing.contains(j.day) else "（上市期间外）")
                for j in s.corrected_factor_jumps
            )
            lines.append(f"- {s.symbol}：{jumps}")
        else:
            lines.append(f"- {s.symbol}：无")
    return lines
