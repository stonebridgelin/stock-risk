"""用例：输出研究样本全部代码的三个区间、端点类型与观察区间内的缺价日（I2、I3、I10）。

经统一取数入口（I12）读取，因此同时验证了全部文件能通过封存检查、量化与明确失败检查。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from stock_risk.application.market_data_loader import MarketDataLoader
from stock_risk.application.ports import UniverseProvider
from stock_risk.domain.observation import SymbolRanges
from stock_risk.domain.universe import Role


@dataclass(frozen=True, slots=True)
class RangeRow:
    symbol: str
    role: Role
    ranges: SymbolRanges
    observed_bars: int
    missing_days: tuple[date, ...]


class RangesWriter(Protocol):
    def write(self, rows: list[RangeRow], out_dir: Path) -> list[Path]: ...


class ShowRangesService:
    def __init__(
        self, loader: MarketDataLoader, universe: UniverseProvider, writer: RangesWriter
    ) -> None:
        self._loader = loader
        self._universe = universe
        self._writer = writer

    def run(self, run_date: date, out_root: Path) -> tuple[list[RangeRow], list[Path]]:
        rows: list[RangeRow] = []
        for m in self._universe.load():
            data = self._loader.load(m.symbol)
            rows.append(
                RangeRow(m.symbol, m.role, data.ranges, len(data.history.bars), data.missing_days)
            )
        return rows, self._writer.write(rows, out_root / run_date.isoformat())
