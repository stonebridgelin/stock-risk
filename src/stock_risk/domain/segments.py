"""多代码分段拼接（T00 / D1）：一个证券由多个数据源代码按日期分段组成。

例：VRX 到 2018-07-13 用数据源代码 VRX，自 2018-07-16 起用 BHC。
规则：各段日期区间不重叠、按日期排序；拼接结果按日期升序且无重复日期。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from stock_risk.domain.market_data import DailyBar, PriceHistory
from stock_risk.domain.seal import assert_dates_not_sealed


class SegmentError(ValueError):
    """分段配置或拼接结果不合规。"""


@dataclass(frozen=True, slots=True)
class SymbolSegment:
    """一段数据源代码及其适用区间（闭区间）；start / end 为 None 表示不设该端边界。"""

    symbol: str
    source_ticker: str
    start: date | None = None
    end: date | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if not self.symbol or not self.source_ticker:
            raise SegmentError("分段的 symbol 与 source_ticker 不能为空")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise SegmentError(f"{self.symbol}/{self.source_ticker}: 起始日晚于结束日")
        assert_dates_not_sealed(f"{self.symbol}/{self.source_ticker} 分段", self.start, self.end)

    def contains(self, day: date) -> bool:
        return (self.start is None or day >= self.start) and (self.end is None or day <= self.end)


def validate_segments(segments: Sequence[SymbolSegment]) -> tuple[SymbolSegment, ...]:
    """检查同一证券的分段：同一 symbol、按日期排序、互不重叠。

    只有首段可以没有起始日、末段可以没有结束日。
    """
    if not segments:
        raise SegmentError("分段列表为空")
    symbols = {s.symbol for s in segments}
    if len(symbols) != 1:
        raise SegmentError(f"分段属于多个证券: {sorted(symbols)}")
    for i, (prev, nxt) in enumerate(zip(segments, segments[1:], strict=False)):
        if prev.end is None or nxt.start is None:
            raise SegmentError(f"{prev.symbol}: 第 {i + 1}、{i + 2} 段之间缺少边界日期")
        if nxt.start <= prev.end:
            raise SegmentError(
                f"{prev.symbol}: 分段重叠或未按日期排序（{prev.end} 与 {nxt.start}）"
            )
    return tuple(segments)


def clip_history(history: PriceHistory, segment: SymbolSegment) -> PriceHistory:
    """只保留落在该段区间内的日线与公司行动。"""
    return PriceHistory(
        bars=tuple(b for b in history.bars if segment.contains(b.day)),
        actions=tuple(a for a in history.actions if segment.contains(a.day)),
    )


def stitch(parts: Sequence[tuple[SymbolSegment, PriceHistory]]) -> PriceHistory:
    """按分段截取后拼接。结果按日期升序且无重复日期，否则抛 SegmentError。"""
    validate_segments([seg for seg, _ in parts])
    bars: list[DailyBar] = []
    actions = []
    for seg, history in parts:
        clipped = clip_history(history, seg)
        bars.extend(clipped.bars)
        actions.extend(clipped.actions)
    days = [b.day for b in bars]
    if len(set(days)) != len(days):
        raise SegmentError("拼接后出现重复日期")
    if days != sorted(days):
        raise SegmentError("拼接后日期未按升序排列")
    actions.sort(key=lambda a: (a.day, a.type))
    return PriceHistory(tuple(bars), tuple(actions))


@dataclass(frozen=True, slots=True)
class JunctionCheck:
    """拼接点检查：前一段最后一根日线与后一段第一根日线。比值只报告，不判定好坏。"""

    symbol: str
    prev_ticker: str
    next_ticker: str
    prev_day: date | None
    next_day: date | None
    prev_close: float | None
    next_open: float | None
    open_to_close_ratio: float | None
    prev_adj_factor: float | None
    next_adj_factor: float | None
    adj_factor_change: float | None


def _factor(b: DailyBar) -> float:
    return b.adj_close / b.close


def junction_checks(
    bars: Sequence[DailyBar], segments: Sequence[SymbolSegment]
) -> list[JunctionCheck]:
    """逐个拼接点检查：后段首开 / 前段末收，以及复权系数（adj_close / close）的变化。"""
    validate_segments(segments)
    out: list[JunctionCheck] = []
    for prev_seg, next_seg in zip(segments, segments[1:], strict=False):
        before = [b for b in bars if prev_seg.contains(b.day)]
        after = [b for b in bars if next_seg.contains(b.day)]
        p = before[-1] if before else None
        n = after[0] if after else None
        pf = _factor(p) if p and p.close > 0 else None
        nf = _factor(n) if n and n.close > 0 else None
        out.append(
            JunctionCheck(
                symbol=prev_seg.symbol,
                prev_ticker=prev_seg.source_ticker,
                next_ticker=next_seg.source_ticker,
                prev_day=p.day if p else None,
                next_day=n.day if n else None,
                prev_close=p.close if p else None,
                next_open=n.open if n else None,
                open_to_close_ratio=n.open / p.close if p and n and p.close > 0 else None,
                prev_adj_factor=pf,
                next_adj_factor=nf,
                adj_factor_change=nf / pf - 1 if pf and nf else None,
            )
        )
    return out
