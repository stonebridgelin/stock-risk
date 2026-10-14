"""check-calendar 与 show-ranges 的输出（CSV、JSON 与 Markdown 摘要）。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from stock_risk.application.check_calendar import CalendarCheckResult
from stock_risk.application.show_ranges import RangeRow


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([["" if v is None else str(v) for v in r] for r in rows])
    return path


class CalendarCheckFileWriter:
    def write(self, result: CalendarCheckResult, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "range": [result.start.isoformat(), result.end.isoformat()],
            "proxy": {
                "symbol": result.proxy_symbol,
                "file_sha256": result.proxy_file_sha256,
                "days": result.proxy_days,
            },
            "reference": {
                "library": result.reference_library,
                "version": result.reference_version,
                "calendar": result.reference_calendar,
                "days": result.reference_days,
            },
            "only_in_proxy": [d.isoformat() for d in result.only_in_proxy],
            "only_in_reference": [d.isoformat() for d in result.only_in_reference],
            "identical": result.identical,
        }
        j = out_dir / "calendar_check.json"
        j.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")
        md = out_dir / "calendar_check.md"
        md.write_text(render_calendar_check(result), "utf-8")
        return [j, md]


def render_calendar_check(r: CalendarCheckResult) -> str:
    def days(ds: tuple) -> str:
        return "、".join(d.isoformat() for d in ds) if ds else "无"

    verdict = "完全一致" if r.identical else "**存在差异（停止项，交产品经理裁定）**"
    return "\n".join(
        [
            "# 交易日历核对（I13）",
            "",
            f"- 比较区间：{r.start} ~ {r.end}",
            f"- 代理：{r.proxy_symbol} 交易日 {r.proxy_days} 天；"
            f"文件 SHA-256 `{r.proxy_file_sha256}`",
            f"- 参照：{r.reference_library} {r.reference_version} 的 {r.reference_calendar}，"
            f"交易日 {r.reference_days} 天",
            f"- 只在代理中：{days(r.only_in_proxy)}",
            f"- 只在参照中：{days(r.only_in_reference)}",
            f"- 结论：{verdict}",
            "",
        ]
    )


class RangesFileWriter:
    def write(self, rows: list[RangeRow], out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        ranges = _write_csv(
            out_dir / "ranges.csv",
            [
                "symbol",
                "role",
                "coverage_start",
                "coverage_end",
                "observation_start",
                "observation_end",
                "replay_start",
                "replay_end",
                "left_end",
                "right_end",
                "left_coverage_verified",
                "observed_bars",
                "missing_days",
            ],
            [
                [
                    r.symbol,
                    r.role,
                    r.ranges.coverage_start,
                    r.ranges.coverage_end,
                    r.ranges.observation_start,
                    r.ranges.observation_end,
                    r.ranges.replay_start,
                    r.ranges.replay_end,
                    r.ranges.left_end,
                    r.ranges.right_end,
                    r.ranges.left_coverage_verified,
                    r.observed_bars,
                    len(r.missing_days),
                ]
                for r in rows
            ],
        )
        missing = _write_csv(
            out_dir / "missing_days.csv",
            ["symbol", "date"],
            [[r.symbol, d] for r in rows for d in r.missing_days],
        )
        md = out_dir / "summary.md"
        md.write_text(render_ranges(rows), "utf-8")
        return [ranges, missing, md]


def render_ranges(rows: list[RangeRow]) -> str:
    lines = [
        "# 三个区间与缺价（I2、I3、I10）",
        "",
        "左端类型为 first_available 的代码左端覆盖未核实。缺价日只统计观察区间内。",
        "",
        "| 代码 | 角色 | 数据覆盖 | 观察区间 | 回放区间 | 左端 | 右端 | 缺价日 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        g = r.ranges
        miss = "、".join(d.isoformat() for d in r.missing_days) if r.missing_days else "0"
        lines.append(
            f"| {r.symbol} | {r.role} | {g.coverage_start} ~ {g.coverage_end} | "
            f"{g.observation_start} ~ {g.observation_end} | {g.replay_start} ~ {g.replay_end} | "
            f"{g.left_end} | {g.right_end} | {miss} |"
        )
    return "\n".join(lines) + "\n"
