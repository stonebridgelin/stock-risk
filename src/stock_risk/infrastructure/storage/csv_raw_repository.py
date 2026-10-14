"""原始行情的 CSV 存储（实现 application.ports.RawDataRepository）。

目录 data/raw/：
  <SYMBOL>.csv            日线（不复权 + 复权）
  <SYMBOL>.actions.csv    该代码的公司行动
  corporate_actions.csv   全部公司行动合并
  _fetch_status.json      每个代码的下载状态（断点续传依据）
  README.txt              来源、时间、口径、成功/缺失清单
  MANIFEST.sha256         日线文件校验值
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from stock_risk.application.models import FetchOutcome, FetchReport
from stock_risk.domain.market_data import PriceHistory
from stock_risk.domain.seal import EXPLORATION_END, VALIDATION_START
from stock_risk.domain.universe import UniverseMember

BAR_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
]
ACTION_COLUMNS = ["date", "type", "value"]
STATUS_FILE = "_fetch_status.json"


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    tmp.replace(path)


def _jsonable(outcome: FetchOutcome) -> dict[str, Any]:
    d = asdict(outcome)
    for k in ("first_day", "last_day"):
        if isinstance(d[k], date):
            d[k] = d[k].isoformat()
    d["warnings"] = list(d["warnings"])
    d.pop("skipped", None)
    for seg in d["segments"]:
        for k in ("first_day", "last_day"):
            if isinstance(seg[k], date):
                seg[k] = seg[k].isoformat()
    if not d["segments"]:
        d.pop("segments")
    return d


def _segments_text(r: dict[str, Any]) -> str:
    segs = r.get("segments") or []
    if not segs:
        return ""
    parts = [
        f"{s['source_ticker']} {s['first_day']} ~ {s['last_day']}（{s['rows']} 行）" for s in segs
    ]
    return "；分段拼接: " + "，".join(parts)


class CsvRawDataRepository:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._status_path = root / STATUS_FILE
        self._status: dict[str, dict[str, Any]] = (
            json.loads(self._status_path.read_text("utf-8")) if self._status_path.exists() else {}
        )

    # ---- RawDataRepository ----
    def is_fetched(self, symbol: str) -> bool:
        return bool(self._status.get(symbol, {}).get("ok")) and self._bar_path(symbol).exists()

    def save(self, member: UniverseMember, history: PriceHistory, outcome: FetchOutcome) -> None:
        _write_csv(
            self._bar_path(member.symbol),
            BAR_COLUMNS,
            [
                [
                    b.day.isoformat(),
                    b.open,
                    b.high,
                    b.low,
                    b.close,
                    b.volume,
                    b.adj_open,
                    b.adj_high,
                    b.adj_low,
                    b.adj_close,
                ]
                for b in history.bars
            ],
        )
        _write_csv(
            self._root / f"{member.symbol}.actions.csv",
            ACTION_COLUMNS,
            [[a.day.isoformat(), a.type.value, a.value] for a in history.actions],
        )
        record = _jsonable(outcome)
        record["sha256"] = hashlib.sha256(self._bar_path(member.symbol).read_bytes()).hexdigest()
        self._status[member.symbol] = record
        self._flush_status()

    def record_failure(self, outcome: FetchOutcome) -> None:
        self._status[outcome.symbol] = _jsonable(outcome)
        self._flush_status()

    def write_summary(self, report: FetchReport) -> None:
        ok = {s: r for s, r in sorted(self._status.items()) if r.get("ok")}
        bad = {s: r for s, r in sorted(self._status.items()) if not r.get("ok")}
        self._write_merged_actions(ok)
        self._write_manifest(ok)
        self._write_readme(report, ok, bad)

    # ---- 内部 ----
    def _bar_path(self, symbol: str) -> Path:
        return self._root / f"{symbol}.csv"

    def _flush_status(self) -> None:
        tmp = self._status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._status, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(self._status_path)

    def _write_merged_actions(self, ok: dict[str, dict[str, Any]]) -> None:
        rows: list[list[Any]] = []
        for symbol in ok:
            p = self._root / f"{symbol}.actions.csv"
            if p.exists():
                with p.open(encoding="utf-8") as f:
                    rows.extend(
                        [symbol, r["date"], r["type"], r["value"]] for r in csv.DictReader(f)
                    )
        _write_csv(self._root / "corporate_actions.csv", ["symbol", *ACTION_COLUMNS], rows)

    def _write_manifest(self, ok: dict[str, dict[str, Any]]) -> None:
        lines = [f"{r['sha256']}  {s}.csv" for s, r in ok.items() if r.get("sha256")]
        (self._root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", "utf-8")

    def _write_readme(
        self, report: FetchReport, ok: dict[str, dict[str, Any]], bad: dict[str, dict[str, Any]]
    ) -> None:
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            "stock-risk 探索期原始行情",
            f"来源: {report.source_name} EOD API",
            f"最近一次运行: {now}",
            f"范围: {report.start} ~ {report.end}"
            f"（{VALIDATION_START} 起为封存验证期，未下载；截止日 {EXPLORATION_END}）",
            "列: open/high/low/close/volume = 不复权原始值；adj_* = 数据源复权值（拆股 + 分红）",
            "公司行动: 来自数据源 divCash / splitFactor；ADR 比例变化数据源不提供",
            "",
            f"成功 {len(ok)} 个:",
        ]
        for s, r in ok.items():
            alias = (
                ""
                if r.get("source_ticker") == s or r.get("segments")
                else f"（数据源代码 {r.get('source_ticker')}）"
            )
            warn = f"，质量提示 {len(r.get('warnings', []))} 条" if r.get("warnings") else ""
            lines.append(
                f"  {s}{alias} [{r.get('category')}]: {r.get('rows')} 行 "
                f"{r.get('first_day')} ~ {r.get('last_day')}，公司行动 {r.get('actions')} 条{warn}"
                f"{_segments_text(r)}"
            )
        lines += ["", f"缺失 {len(bad)} 个:"]
        lines += [f"  {s} [{r.get('category')}]: {r.get('error')}" for s, r in bad.items()]
        (self._root / "README.txt").write_text("\n".join(lines) + "\n", "utf-8")
