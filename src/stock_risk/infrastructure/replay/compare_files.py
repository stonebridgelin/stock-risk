"""T06 对照回放的文件读写（实现 application.compare.service.CompareOutputSink）。

目录 output/compare/<run_id>/：
  run_manifest.json、plotly.min.js、各汇总表
  （coverage_summary.csv、pairs_summary.csv、burden.csv 等）
  sets/<套>/events.csv、episodes.csv、daily_compact.csv.gz、
            coverage_daily_<套>.csv.gz、event_compare_<套>.csv、overlap_<套>.csv（新变体）
  sets/<套>/full/<对象>/daily.csv、chart.html（只有时间线用的两套）
  timelines/<代码>_<年份>.csv、index.html
gzip 用标准库；压缩级别 6。本类只保存根目录，可以传给子进程。
"""

from __future__ import annotations

import csv
import gzip
import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from stock_risk.application.replay_chart_data import ChartData
from stock_risk.application.replay_outputs import DAILY_COLUMNS
from stock_risk.infrastructure.replay.chart import PLOTLY_JS, render_chart, write_plotly_js
from stock_risk.infrastructure.replay.files import _fmt


class CompareFileSink:
    def __init__(self, root: Path) -> None:
        self.root = root

    def create_run_dir(self, run_id: str) -> Path:
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=False)
        return path

    def run_dir_exists(self, run_id: str) -> bool:
        return (self.root / run_id).exists()

    def set_dir(self, run_dir: Path, set_id: str) -> Path:
        d = run_dir / "sets" / set_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_csv(
        self, path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
    ) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
            w.writeheader()
            w.writerows({k: _fmt(r.get(k, "")) for k in columns} for r in rows)

    def write_csv_gz(
        self, path: Path, columns: Sequence[str], rows: Sequence[Sequence[Any]]
    ) -> None:
        with gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=6) as f:
            w = csv.writer(f)
            w.writerow(columns)
            w.writerows([_fmt(v) for v in r] for r in rows)

    def read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def write_full(
        self,
        run_dir: Path,
        set_id: str,
        target: str,
        daily: Sequence[Mapping[str, Any]],
        chart: ChartData,
    ) -> None:
        d = self.set_dir(run_dir, set_id) / "full" / target
        d.mkdir(parents=True, exist_ok=True)
        self.write_csv(d / "daily.csv", DAILY_COLUMNS, daily)
        render_chart(chart, d / "chart.html", f"../../../../{PLOTLY_JS}")

    def write_plotly_js(self, run_dir: Path) -> None:
        write_plotly_js(run_dir)

    def write_manifest(self, run_dir: Path, manifest: Mapping[str, Any]) -> Path:
        path = run_dir / "run_manifest.json"
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", "utf-8"
        )
        return path

    def write_timeline_index(
        self, run_dir: Path, run_id: str, files: Sequence[Mapping[str, Any]]
    ) -> Path:
        """timelines/index.html：两套的通知逐条并排，附两套完整 chart.html 的链接。离线可看。"""
        parts = []
        for f in files:
            sets = f["sets"]
            head = f"<h2>{html.escape(f['target'])} {f['year']}</h2>"
            links = " · ".join(
                f'<a href="../sets/{html.escape(s)}/full/{html.escape(f["target"])}/chart.html">'
                f"{html.escape(s)} 图表</a>"
                for s in sets
            )
            cells = []
            for s in sets:
                rows = [r for r in f["rows"] if r["set_id"] == s]
                lis = "".join(
                    f"<tr><td>{html.escape(str(r.get('date', '')))}</td>"
                    f"<td>{html.escape(str(r.get('event_type') or r.get('kind')))}</td>"
                    f"<td>{html.escape(str(r.get('level_before', '')))} → "
                    f"{html.escape(str(r.get('level_after', '')))}</td>"
                    f"<td>{html.escape(str(r.get('reasons', '') or r.get('detail', '')))}</td>"
                    f"<td>{html.escape(str(_fmt(r.get('close', ''))))}</td></tr>"
                    for r in rows
                )
                cells.append(
                    f"<td style='vertical-align:top'><h3>{html.escape(s)}</h3>"
                    f"<table><tr><th>日期</th><th>事件</th><th>等级前后</th><th>原因</th>"
                    f"<th>收盘</th></tr>{lis}</table></td>"
                )
            parts.append(
                f"{head}<p>{links} · <a href='{html.escape(f['file'])}'>CSV</a></p>"
                f"<table><tr>{''.join(cells)}</tr></table>"
            )
        page = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>时间线 {html.escape(run_id)}</title>
<style>body{{font-family:sans-serif;margin:16px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:3px 6px;font-size:13px}}</style></head><body>
<h1>单股年度通知时间线（{html.escape(run_id)}）</h1>
<p>取自连续回放，不重新初始化；每段首行为年初状态。只陈述通知，不含评价或结论。</p>
{"".join(parts)}</body></html>
"""
        path = run_dir / "timelines" / "index.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page, "utf-8")
        return path
