"""Massive 付费档验证的进度与输出（实现 application.probe_massive_paid.PaidStore），
目录 data/massive_probe/paid/（不入库）：

_progress.json                 各项状态与计数（断点续传依据；不含原始响应）
aggs/<代码>_<raw|adj>.csv        第 1 节日线：date,open,high,low,close,volume,vwap,trades
actions/<代码>_splits.csv        第 2 节拆股：execution_date,split_from,split_to,adjustment_type
actions/<代码>_dividends.csv     第 2 节分红：除息日、金额、币种、其他日期（晚于截止日的已写为空）等
lists/lists_<快照>.csv           第 3 节清单逐行（白名单字段）
lists/labels_<快照>.csv          第 3 节逐行补标签结果与所用规则
samples.csv                    第 3 节抽样
summary.json                   汇总

另提供交易日历（data/raw/SPY.csv）与 Tiingo 原始价、公司行动（data/raw_selection/）的读取。
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from stock_risk.domain.massive_labels import LIST_FIELDS
from stock_risk.domain.massive_paid import (
    Bar,
    Dividend,
    ReferenceAction,
    ReferenceBar,
    Split,
)
from stock_risk.infrastructure.storage.massive_probe_files import MassiveProbeFiles, write_csv

BAR_COLUMNS = ["date", "open", "high", "low", "close", "volume", "vwap", "trades"]
SPLIT_COLUMNS = ["execution_date", "split_from", "split_to", "adjustment_type"]
DIVIDEND_COLUMNS = [
    "ex_dividend_date",
    "cash_amount",
    "currency",
    "declaration_date",
    "record_date",
    "pay_date",
    "frequency",
    "distribution_type",
    "masked_after_2019",
]
LIST_COLUMNS = [*LIST_FIELDS, "list_date", "delisted_utc", "delisted_after_2019"]
LABEL_COLUMNS = [
    "ticker",
    "cik",
    "composite_figi",
    "primary_exchange",
    "exchange_filled",
    "exchange_rule",
    "exchange_conflict",
    "type",
    "type_filled",
    "type_rule",
    "type_conflict",
]
SAMPLE_COLUMNS = [
    "group",
    "date",
    "snapshot",
    "ticker",
    "cik",
    "composite_figi",
    "overview_status",
    "sic_code",
    "overview_type",
    "overview_primary_exchange",
    "prices_status",
    "first",
    "last",
    "rows",
]


def _iso(d: date | None) -> str | None:
    return d.isoformat() if d else None


def _opt_date(v: str) -> date | None:
    return date.fromisoformat(v) if v else None


def _opt_float(v: str) -> float | None:
    return float(v) if v != "" else None


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class MassivePaidFiles:
    def __init__(self, out_dir: Path) -> None:
        self._dir = out_dir
        self._progress = MassiveProbeFiles(out_dir)  # 进度文件的读写与 T08d-2a 相同
        for sub in ("aggs", "actions", "lists"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)

    def load_progress(self) -> dict[str, Any]:
        return self._progress.load_progress()

    def save_progress(self, progress: Mapping[str, Any]) -> None:
        self._progress.save_progress(progress)

    # ---- 日线 ----

    def _bars_path(self, ticker: str, kind: str) -> Path:
        return self._dir / "aggs" / f"{ticker}_{kind}.csv"

    def save_bars(self, ticker: str, kind: str, bars: Sequence[Bar]) -> None:
        rows = [
            {
                "date": b.day.isoformat(),
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "vwap": b.vwap,
                "trades": b.trades,
            }
            for b in bars
        ]
        write_csv(self._bars_path(ticker, kind), BAR_COLUMNS, rows)

    def load_bars(self, ticker: str, kind: str) -> list[Bar]:
        return [
            Bar(
                date.fromisoformat(r["date"]),
                _opt_float(r["open"]),
                _opt_float(r["high"]),
                _opt_float(r["low"]),
                _opt_float(r["close"]),
                _opt_float(r["volume"]),
                _opt_float(r["vwap"]),
                int(r["trades"]) if r["trades"] else None,
            )
            for r in _read(self._bars_path(ticker, kind))
        ]

    # ---- 拆股与分红 ----

    def save_actions(
        self, ticker: str, splits: Sequence[Split], dividends: Sequence[Dividend]
    ) -> None:
        write_csv(
            self._dir / "actions" / f"{ticker}_splits.csv",
            SPLIT_COLUMNS,
            [
                {
                    "execution_date": s.day.isoformat(),
                    "split_from": s.split_from,
                    "split_to": s.split_to,
                    "adjustment_type": s.adjustment_type,
                }
                for s in splits
            ],
        )
        write_csv(
            self._dir / "actions" / f"{ticker}_dividends.csv",
            DIVIDEND_COLUMNS,
            [
                {
                    "ex_dividend_date": d.day.isoformat(),
                    "cash_amount": d.cash_amount,
                    "currency": d.currency,
                    "declaration_date": _iso(d.declaration_date),
                    "record_date": _iso(d.record_date),
                    "pay_date": _iso(d.pay_date),
                    "frequency": d.frequency,
                    "distribution_type": d.distribution_type,
                    "masked_after_2019": "|".join(d.masked_after_2019),
                }
                for d in dividends
            ],
        )

    def load_actions(self, ticker: str) -> tuple[list[Split], list[Dividend]]:
        splits = [
            Split(
                date.fromisoformat(r["execution_date"]),
                float(r["split_from"]),
                float(r["split_to"]),
                r["adjustment_type"] or None,
            )
            for r in _read(self._dir / "actions" / f"{ticker}_splits.csv")
        ]
        divs = [
            Dividend(
                date.fromisoformat(r["ex_dividend_date"]),
                _opt_float(r["cash_amount"]),
                r["currency"] or None,
                _opt_date(r["declaration_date"]),
                _opt_date(r["record_date"]),
                _opt_date(r["pay_date"]),
                int(r["frequency"]) if r["frequency"] else None,
                r["distribution_type"] or None,
                tuple(x for x in r["masked_after_2019"].split("|") if x),
            )
            for r in _read(self._dir / "actions" / f"{ticker}_dividends.csv")
        ]
        return splits, divs

    # ---- 清单 ----

    def save_list(self, key: str, rows: Sequence[Mapping[str, Any]]) -> None:
        write_csv(self._dir / "lists" / f"lists_{key}.csv", LIST_COLUMNS, rows)

    def load_list(self, key: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in _read(self._dir / "lists" / f"lists_{key}.csv"):
            row: dict[str, Any] = {k: (v if v != "" else None) for k, v in r.items()}
            row["active"] = r["active"] == "true" if r["active"] else None
            row["delisted_after_2019"] = r["delisted_after_2019"] == "true"
            out.append(row)
        return out

    # ---- 汇总 ----

    def write_outputs(
        self,
        labels: Mapping[str, Sequence[Mapping[str, Any]]],
        samples: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> list[str]:
        paths: list[Path] = []
        for key, rows in sorted(labels.items()):
            p = self._dir / "lists" / f"labels_{key}.csv"
            write_csv(p, LABEL_COLUMNS, rows)
            paths.append(p)
        p = self._dir / "samples.csv"
        write_csv(p, SAMPLE_COLUMNS, samples)
        paths.append(p)
        p = self._dir / "summary.json"
        p.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8")
        paths.append(p)
        return [str(x) for x in paths]


# ---- 读取对照数据（只读，不发请求） ----


def read_calendar(spy_csv: Path) -> list[date]:
    """交易日历：data/raw/SPY.csv 的日期列。"""
    return [date.fromisoformat(r["date"]) for r in _read(spy_csv)]


def read_tiingo_prices(raw_dir: Path, tickers: Sequence[str]) -> dict[str, list[ReferenceBar]]:
    """Tiingo 原始价（未复权 close、volume）。文件不存在的代码不列出。"""
    out: dict[str, list[ReferenceBar]] = {}
    for t in tickers:
        p = raw_dir / f"{t}.csv"
        if p.exists():
            out[t] = [
                ReferenceBar(date.fromisoformat(r["date"]), float(r["close"]), float(r["volume"]))
                for r in _read(p)
            ]
    return out


def read_tiingo_actions(raw_dir: Path, tickers: Sequence[str]) -> dict[str, list[ReferenceAction]]:
    """Tiingo 公司行动（date,type,value；dividend 为金额，split 为 splitFactor）。"""
    out: dict[str, list[ReferenceAction]] = {}
    for t in tickers:
        p = raw_dir / f"{t}.actions.csv"
        if p.exists():
            out[t] = [
                ReferenceAction(date.fromisoformat(r["date"]), r["type"], float(r["value"]))
                for r in _read(p)
            ]
    return out
