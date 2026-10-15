"""回放输出的文件读写（实现 application.replay_run.ReplayOutputSink）。

目录 output/replay/<run_id>/：
  run_manifest.json、index.html、summary.csv、summary_by_candidate.csv、plotly.min.js
  c<候选编号>/<对象>/events.csv、daily.csv、episodes.csv、chart.html
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from stock_risk.application.replay_chart_data import ChartData
from stock_risk.application.replay_outputs import DAILY_COLUMNS, EPISODE_COLUMNS, EVENT_COLUMNS
from stock_risk.infrastructure.replay.chart import (
    PLOTLY_JS,
    read_chart_events,
    render_chart,
    write_plotly_js,
)
from stock_risk.infrastructure.replay.csv_io import format_rows, read_csv, write_csv
from stock_risk.infrastructure.replay.csv_io import write_manifest as _write_manifest


class ReplayFileSink:
    def __init__(self, root: Path) -> None:
        self._root = root

    def create_run_dir(self, run_id: str) -> Path:
        path = self._root / run_id
        path.mkdir(parents=True, exist_ok=False)  # 按 run_id 分开，不覆盖
        return path

    def run_dir_exists(self, run_id: str) -> bool:
        return (self._root / run_id).exists()

    def target_dir(self, run_dir: Path, candidate_id: int, label: str) -> Path:
        return run_dir / f"c{candidate_id:03d}" / label

    def write_target(
        self,
        run_dir: Path,
        candidate_id: int,
        label: str,
        events: Sequence[Mapping[str, Any]],
        daily: Sequence[Mapping[str, Any]],
        episodes: Sequence[Mapping[str, Any]],
        chart: ChartData,
    ) -> Path:
        d = self.target_dir(run_dir, candidate_id, label)
        d.mkdir(parents=True, exist_ok=False)
        write_csv(d / "events.csv", EVENT_COLUMNS, events)
        write_csv(d / "daily.csv", DAILY_COLUMNS, daily)
        write_csv(d / "episodes.csv", EPISODE_COLUMNS, episodes)
        write_plotly_js(run_dir)
        render_chart(chart, d / "chart.html", f"../../{PLOTLY_JS}")
        return d

    def read_target(
        self, run_dir: Path, candidate_id: int, label: str
    ) -> tuple[
        list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[tuple[str, str]]
    ]:
        """读回写出的四种输出（一致性检查用）。"""
        d = self.target_dir(run_dir, candidate_id, label)
        return (
            read_csv(d / "events.csv"),
            read_csv(d / "daily.csv"),
            read_csv(d / "episodes.csv"),
            read_chart_events(d / "chart.html"),
        )

    def format_rows(
        self, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, str]]:
        """按写 CSV 时的同一规则格式化后、经过 CSV 往返的结果（V0 回归比较用）。"""
        return format_rows(columns, rows)

    def read_tables(
        self, run_dir: Path, candidate_id: int, label: str
    ) -> dict[str, list[dict[str, str]]]:
        """读取某次运行中一个（候选, 对象）的 events、daily、episodes。"""
        d = self.target_dir(run_dir, candidate_id, label)
        return {name: read_csv(d / f"{name}.csv") for name in ("events", "daily", "episodes")}

    def write_summary(
        self, run_dir: Path, rows: Sequence[Mapping[str, Any]], by_candidate: Sequence[Mapping]
    ) -> list[Path]:
        out = []
        for name, data in (("summary.csv", rows), ("summary_by_candidate.csv", by_candidate)):
            if data:
                cols = [k for k in data[0] if not k.startswith("_")]
                write_csv(run_dir / name, cols, data)
                out.append(run_dir / name)
        return out

    def write_manifest(self, run_dir: Path, manifest: Mapping[str, Any]) -> Path:
        return _write_manifest(run_dir, manifest)

    def write_index(
        self, run_dir: Path, run_id: str, run_kind: str, cells: Sequence[Mapping[str, Any]]
    ) -> Path:
        """index.html：按个股 × 候选列出全部图表链接，并附预警次数。

        cells：[{target, candidate_id, label, episodes}]。
        """
        targets = sorted({c["target"] for c in cells})
        cands = sorted({(c["candidate_id"], c["label"]) for c in cells})
        by = {(c["target"], c["candidate_id"]): c for c in cells}
        head = "".join(
            f"<th>c{cid:03d}<br><small>{html.escape(lab)}</small></th>" for cid, lab in cands
        )
        rows = []
        for t in targets:
            tds = []
            for cid, _ in cands:
                c = by.get((t, cid))
                if c is None:
                    tds.append("<td>—</td>")
                    continue
                href = f"c{cid:03d}/{html.escape(t)}/chart.html"
                tds.append(f'<td><a href="{href}">{c["episodes"]} 次</a></td>')
            rows.append(f"<tr><th>{html.escape(t)}</th>{''.join(tds)}</tr>")
        page = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>回放 {html.escape(run_id)}</title>
<style>body{{font-family:sans-serif;margin:16px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 6px;text-align:center;font-size:13px}}
th small{{font-weight:normal;color:#555}}</style></head><body>
<h1>回放 {html.escape(run_id)}（{html.escape(run_kind)}）</h1>
<p>单元格为该个股在该候选下的预警次数（全部启动的预警，含 2009 年预热期），点击打开图表。
运行清单见 <a href="run_manifest.json">run_manifest.json</a>，汇总见 summary.csv。
本页只陈述回放结果，不含评价或结论。</p>
<table><tr><th>个股</th>{head}</tr>{"".join(rows)}</table></body></html>
"""
        path = run_dir / "index.html"
        path.write_text(page, "utf-8")
        return path
