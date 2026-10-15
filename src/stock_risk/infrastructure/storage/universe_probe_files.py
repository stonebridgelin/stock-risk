"""代码清单探查的输出（实现 application.probe_tickers.ProbeWriter），目录 data/universe_probe/：

  tickers_2008_2019.csv   ticker,exchange,start_date,end_date,active_after_2019
  summary.json            计数
  coverage_check.csv      覆盖对照（给了 --check 时）

另提供覆盖对照输入（--check）的读取。
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from stock_risk.domain.universe_probe import CoverageRequest, TickerRow

TICKER_COLUMNS = ["ticker", "exchange", "start_date", "end_date", "active_after_2019"]
CHECK_COLUMNS = ["ticker", "from", "to", "in_list", "covered", "list_ranges", "other_listing"]


def _cell(v: Any) -> Any:
    if isinstance(v, bool):
        return "true" if v else "false"
    return v


class ProbeFileWriter:
    def write(
        self,
        out_dir: Path,
        rows: Sequence[TickerRow],
        summary: Mapping[str, Any],
        checked: Sequence[Mapping[str, Any]] | None,
    ) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        tickers = out_dir / "tickers_2008_2019.csv"
        with tickers.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(TICKER_COLUMNS)
            w.writerows(
                [
                    r.ticker,
                    r.exchange,
                    r.start.isoformat(),
                    r.end.isoformat(),
                    _cell(r.active_after_2019),
                ]
                for r in rows
            )
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8")
        paths = [tickers, summary_path]
        if checked is not None:
            check_path = out_dir / "coverage_check.csv"
            with check_path.open("w", newline="", encoding="utf-8") as f:
                dw = csv.DictWriter(f, fieldnames=CHECK_COLUMNS)
                dw.writeheader()
                dw.writerows({k: _cell(r[k]) for k in CHECK_COLUMNS} for r in checked)
            paths.append(check_path)
        return paths


def read_coverage_requests(path: Path) -> list[CoverageRequest]:
    """一列 ticker，可附 from、to（ISO 日期，可空）。to 晚于 2019-12-31 时报错。"""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "ticker" not in (reader.fieldnames or []):
            raise ValueError(f"{path}: 缺少 ticker 列")

        def day(text: str | None) -> date | None:
            return date.fromisoformat(text.strip()) if text and text.strip() else None

        return [
            CoverageRequest(r["ticker"].strip(), day(r.get("from")), day(r.get("to")))
            for r in reader
            if r["ticker"].strip()
        ]
