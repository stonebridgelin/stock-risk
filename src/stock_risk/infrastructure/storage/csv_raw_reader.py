"""读取 data/raw 中的原始行情（实现 application.ports.RawDataReader）。

封存纪律：晚于探索期截止日的行只计数，不解析数值、不返回。
"""

from __future__ import annotations

import csv
import hashlib
from datetime import date
from pathlib import Path

from stock_risk.application.ports import RawFile
from stock_risk.domain.audit import count_sealed_rows
from stock_risk.domain.market_data import ActionType, CorporateAction, DailyBar, PriceHistory
from stock_risk.domain.seal import EXPLORATION_END

MANIFEST_FILE = "MANIFEST.sha256"


class CsvRawDataReader:
    def __init__(self, root: Path) -> None:
        self._root = root

    def exists(self, symbol: str) -> bool:
        return (self._root / f"{symbol}.csv").exists()

    def load(self, symbol: str) -> RawFile:
        bars: list[DailyBar] = []
        sealed_days: list[date] = []
        with (self._root / f"{symbol}.csv").open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                d = date.fromisoformat(r["date"])
                if d > EXPLORATION_END:
                    sealed_days.append(d)
                    continue
                bars.append(
                    DailyBar(
                        day=d,
                        open=float(r["open"]),
                        high=float(r["high"]),
                        low=float(r["low"]),
                        close=float(r["close"]),
                        volume=float(r["volume"]),
                        adj_open=float(r["adj_open"]),
                        adj_high=float(r["adj_high"]),
                        adj_low=float(r["adj_low"]),
                        adj_close=float(r["adj_close"]),
                    )
                )
        days = [b.day for b in bars]
        if any(a >= b for a, b in zip(days, days[1:], strict=False)):
            raise ValueError(f"{symbol}.csv 日期未严格升序（重复或乱序）")
        actions: list[CorporateAction] = []
        action_path = self._root / f"{symbol}.actions.csv"
        if action_path.exists():
            with action_path.open(encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    d = date.fromisoformat(r["date"])
                    if d > EXPLORATION_END:
                        sealed_days.append(d)
                        continue
                    actions.append(CorporateAction(d, ActionType(r["type"]), float(r["value"])))
        return RawFile(
            symbol=symbol,
            history=PriceHistory(tuple(bars), tuple(actions)),
            sealed_rows=count_sealed_rows(sealed_days),
        )

    def manifest_sha256(self) -> str | None:
        p = self._root / MANIFEST_FILE
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
