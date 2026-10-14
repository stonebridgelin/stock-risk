"""用例：单股串行回放（T04a）。

- 唯一取数入口为 MarketDataLoader（I12），不自行读 data/raw。
- 按 T01 的三个区间与逐日序列运行：引擎从回放起点开始（2008 年只作预热），到观察终点调用截尾（I3）。
- 按 G 切分历史段，每个历史段一个指标实现（T02 的纯函数）。
- 只回放 role = signal_target 的个股（I15）。
- 登记了复权价更正的代码（SHLD）同时回放更正口径与原口径（I11），分别标 corrected / uncorrected。
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from datetime import date

from stock_risk.application.market_data_loader import MarketDataLoader, ObservedData
from stock_risk.application.ports import UniverseProvider
from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.indicator_source import ComputedIndicators
from stock_risk.domain.lifecycle.model import EndReason
from stock_risk.domain.lifecycle.params import EngineParams
from stock_risk.domain.lifecycle.runner import RunResult, build_sources, run_bars
from stock_risk.domain.observation import RightEnd
from stock_risk.domain.universe import Role

CORRECTED, UNCORRECTED = "corrected", "uncorrected"
END_REASON = {
    RightEnd.LISTING_END: EndReason.LISTING_END,
    RightEnd.EXPLORATION_END: EndReason.EXPLORATION_END,
}


@dataclass(frozen=True, slots=True)
class ReplayTarget:
    """回放对象：代码与口径。variant 为空表示该代码没有登记更正，只有一套结果。"""

    symbol: str
    variant: str = ""

    @property
    def corrected(self) -> bool:
        return self.variant != UNCORRECTED

    @property
    def label(self) -> str:
        return self.symbol if not self.variant else f"{self.symbol}_{self.variant}"


@dataclass(frozen=True, slots=True)
class PreparedSymbol:
    """一个回放对象的输入：取数入口的结果与各历史段的指标（多个候选共用）。"""

    target: ReplayTarget
    data: ObservedData
    gap_break: int
    sources: tuple[ComputedIndicators, ...]


@dataclass(frozen=True, slots=True)
class SymbolReplay:
    target: ReplayTarget
    candidate_id: int
    params: EngineParams
    prepared: PreparedSymbol
    result: RunResult


class ReplayService:
    def __init__(self, loader: MarketDataLoader, universe: UniverseProvider) -> None:
        self._loader = loader
        self._universe = universe

    @property
    def loader(self) -> MarketDataLoader:
        return self._loader

    def targets(self, symbols: Collection[str] | None = None) -> list[ReplayTarget]:
        """研究样本中 role = signal_target 的个股；登记了更正的代码给出两套口径。"""
        members = [m for m in self._universe.load() if m.role is Role.SIGNAL_TARGET]
        if symbols is not None:
            wanted = {s.upper() for s in symbols}
            unknown = wanted - {m.symbol for m in members}
            if unknown:
                raise ValueError(f"不是研究样本中的个股（signal_target）: {sorted(unknown)}")
            members = [m for m in members if m.symbol in wanted]
        out: list[ReplayTarget] = []
        for m in members:
            if m.symbol in self._loader.corrections:
                out += [ReplayTarget(m.symbol, CORRECTED), ReplayTarget(m.symbol, UNCORRECTED)]
            else:
                out.append(ReplayTarget(m.symbol))
        return out

    def prepare(self, target: ReplayTarget, gap_break: int, cache: bool = True) -> PreparedSymbol:
        data = self._loader.load(target.symbol, corrected=target.corrected)
        sources = build_sources(
            self._loader.calendar(), data.presence, data.history.bars, gap_break, cache=cache
        )
        return PreparedSymbol(target, data, gap_break, tuple(sources))

    def replay(
        self, prepared: PreparedSymbol, candidate_id: int, params: EngineParams
    ) -> SymbolReplay:
        return SymbolReplay(
            prepared.target,
            candidate_id,
            params,
            prepared,
            replay_prepared(prepared, params, self._loader.calendar()),
        )


def replay_prepared(
    prepared: PreparedSymbol,
    params: EngineParams,
    calendar: TradingCalendar,
    upto: date | None = None,
) -> RunResult:
    """回放一个已准备的对象。upto 给定时只回放到该日（无前视检查用），不做截尾。"""
    if params.assumptions.gap_break != prepared.gap_break:
        raise ValueError("准备数据时的 G 与参数中的 G 不同")
    data = prepared.data
    if upto is None:
        return run_bars(
            prepared.target.symbol,
            calendar,
            data.presence,
            data.history.bars,
            params,
            data.ranges.replay_start,
            END_REASON[data.ranges.right_end],
            list(prepared.sources),
        )
    presence = [p for p in data.presence if p.day <= upto]
    bars = [b for b in data.history.bars if b.day <= upto]
    return run_bars(
        prepared.target.symbol,
        calendar,
        presence,
        bars,
        params,
        data.ranges.replay_start,
        None,
        build_sources(calendar, presence, bars, prepared.gap_break, cache=False),
    )
