"""D0b P1：证券身份、映射一致性与缺口清单（T12-D0b 第 2.1、2.2、2.7 节）。纯计算，不联网。

- 构建器的映射（resolve_segments）：config/massive_source_tickers.csv 中登记的证券用它的分段
  （PM-D0b-01）；否则用 config/symbol_segments.csv；都没有的，源代码 = 研究样本代码。
  fetch-massive-d0b 与 build-canonical 都经这一个函数。
- 证券身份（determine_identity）由本地证据独立认定：中转服务核验（verify/prices_<T>.csv）
  按研究样本代码取到的行数与起止日。规则：
  ① 中转服务有数据，且末日与 Tiingo 末日相同或由上市期间解释 → 整段用研究样本代码；
  ② 中转服务末日早于 Tiingo 末日、又不能由上市期间解释 → 后一段用 source_candidates 中的
     下一个代码，自中转末日之后的第一个交易日起；没有下一个候选即为疑问；
  ③ 中转服务 0 行 → 只有在源代码映射配置中有带裁定编号的登记时才采用，否则为疑问。
  有疑问的抛 IdentityQuestion，停止（第 2.1 节）。
- 范围：effective_range = [max(2008-01-01, 上市起), min(2019-12-31, 上市止)]；每段再与分段交集。
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from stock_risk.application.fetch_massive_d0b import CATEGORIES, RequestSpec
from stock_risk.domain.listing import ListingPeriod
from stock_risk.domain.seal import EXPLORATION_END
from stock_risk.domain.segments import SymbolSegment
from stock_risk.domain.universe import UniverseMember

RANGE_START = date(2008, 1, 1)
RANGE_END = EXPLORATION_END


class IdentityQuestion(RuntimeError):
    """证券身份或代码选择有疑问：停止，提交裁决（第 2.1 节）。"""


def resolve_segments(
    symbol: str,
    symbol_segments: Mapping[str, tuple[SymbolSegment, ...]],
    massive_tickers: Mapping[str, tuple[SymbolSegment, ...]],
) -> tuple[SymbolSegment, ...]:
    """构建器实际使用的 Massive 源代码分段。"""
    return (
        massive_tickers.get(symbol)
        or symbol_segments.get(symbol)
        or (SymbolSegment(symbol, symbol),)
    )


def effective_range(listing: ListingPeriod | None) -> tuple[date, date]:
    start, end = RANGE_START, RANGE_END
    if listing is not None:
        if listing.listed_from is not None:
            start = max(start, listing.listed_from)
        if listing.listed_until is not None:
            end = min(end, listing.listed_until)
    return start, end


@dataclass(frozen=True, slots=True)
class Segment:
    ticker: str
    start: date
    end: date

    def text(self) -> str:
        return f"{self.ticker}:{self.start.isoformat()}~{self.end.isoformat()}"


def segment_ranges(
    segments: Sequence[SymbolSegment], rng: tuple[date, date]
) -> tuple[Segment, ...]:
    out = []
    for s in segments:
        start = max(rng[0], s.start or rng[0])
        end = min(rng[1], s.end or rng[1])
        if start <= end:
            out.append(Segment(s.source_ticker, start, end))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class VerifyEvidence:
    """verify/prices_<T>.csv 的摘要（2010-01-01 至 2019-12-31 的核验）。"""

    file: str
    relay_rows: int
    relay_first: date | None
    relay_last: date | None
    tiingo_rows: int
    tiingo_last: date | None


@dataclass(frozen=True, slots=True)
class IdentityRow:
    symbol: str
    segments: tuple[Segment, ...]
    listing: str
    range: tuple[date, date]
    evidence: str

    def record(self) -> dict[str, str]:
        return {
            "symbol": self.symbol,
            "segments": "|".join(s.text() for s in self.segments),
            "listing": self.listing,
            "range": f"{self.range[0].isoformat()}~{self.range[1].isoformat()}",
            "evidence": self.evidence,
        }


def _listing_text(p: ListingPeriod | None) -> str:
    if p is None:
        return ""
    a = p.listed_from.isoformat() if p.listed_from else ""
    b = p.listed_until.isoformat() if p.listed_until else ""
    return f"{a}~{b}"


def _next_session(calendar: Sequence[date], d: date) -> date:
    i = bisect.bisect_right(calendar, d)
    if i >= len(calendar):
        raise IdentityQuestion(f"日历中没有 {d} 之后的交易日")
    return calendar[i]


def determine_identity(
    members: Sequence[UniverseMember],
    evidence: Mapping[str, VerifyEvidence],
    listings: Mapping[str, ListingPeriod],
    ruled: Mapping[str, tuple[tuple[SymbolSegment, ...], str]],
    calendar: Sequence[date],
) -> tuple[IdentityRow, ...]:
    """按本地证据独立认定每只证券的源代码与分段（规则见模块说明）。

    ruled：symbol → (分段, 裁定编号)。"""
    rows: list[IdentityRow] = []
    for m in members:
        sym = m.symbol
        listing = listings.get(sym)
        rng = effective_range(listing)
        ev = evidence.get(sym)
        if ev is None:
            raise IdentityQuestion(f"{sym}: 没有中转服务核验文件，无法认定代码")
        base = (
            f"{ev.file}：中转 {ev.relay_rows} 行"
            + (f" {ev.relay_first}~{ev.relay_last}" if ev.relay_rows else "")
            + f"；Tiingo {ev.tiingo_rows} 行（核验范围 2010-01-01~2019-12-31，2008–2009 年未核验）"
        )
        if ev.relay_rows == 0:
            if sym not in ruled:
                raise IdentityQuestion(f"{sym}: 中转服务按 {sym} 返回 0 行，代码选择需要裁决")
            segs, ruling = ruled[sym]
            rows.append(
                IdentityRow(
                    sym,
                    segment_ranges(segs, rng),
                    _listing_text(listing),
                    rng,
                    f"{base}；按裁定 {ruling} 采用 {'、'.join(s.source_ticker for s in segs)}",
                )
            )
            continue
        assert ev.relay_last is not None
        explained = ev.tiingo_last is None or ev.relay_last >= ev.tiingo_last
        if not explained and listing is not None and listing.listed_until is not None:
            explained = ev.relay_last >= listing.listed_until
        if explained:
            note = "末日与 Tiingo 相同" if ev.relay_last == ev.tiingo_last else "末日由上市期间解释"
            rows.append(
                IdentityRow(
                    sym,
                    segment_ranges((SymbolSegment(sym, sym),), rng),
                    _listing_text(listing),
                    rng,
                    f"{base}；{note}，整段用 {sym}",
                )
            )
            continue
        candidates = list(m.source_candidates)
        nxt = candidates[1] if len(candidates) > 1 and candidates[0] == sym else None
        if nxt is None:
            raise IdentityQuestion(
                f"{sym}: 中转服务止于 {ev.relay_last}，早于 Tiingo 末日 {ev.tiingo_last}，"
                "没有下一个候选代码"
            )
        boundary = _next_session(calendar, ev.relay_last)
        segs = (
            SymbolSegment(sym, sym, None, ev.relay_last),
            SymbolSegment(sym, nxt, boundary, None),
        )
        rows.append(
            IdentityRow(
                sym,
                segment_ranges(segs, rng),
                _listing_text(listing),
                rng,
                f"{base}；中转止于 {ev.relay_last} 而 Tiingo 延续，"
                f"后一段用候选代码 {nxt}，自 {boundary} 起（{nxt} 段本地没有核验文件）",
            )
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class MappingCompare:
    symbol: str
    identity: str
    builder: str

    @property
    def same(self) -> bool:
        return self.identity == self.builder


def compare_mapping(
    identity: Sequence[IdentityRow],
    symbol_segments: Mapping[str, tuple[SymbolSegment, ...]],
    massive_tickers: Mapping[str, tuple[SymbolSegment, ...]],
    listings: Mapping[str, ListingPeriod],
) -> tuple[MappingCompare, ...]:
    """第 2.7 节：构建器实际使用的源代码与区间，与身份清单逐行比较。"""
    out = []
    for row in identity:
        segs = resolve_segments(row.symbol, symbol_segments, massive_tickers)
        built = segment_ranges(segs, effective_range(listings.get(row.symbol)))
        out.append(
            MappingCompare(
                row.symbol,
                "|".join(s.text() for s in row.segments),
                "|".join(s.text() for s in built),
            )
        )
    return tuple(out)


def gap_specs(identity: Sequence[IdentityRow]) -> tuple[RequestSpec, ...]:
    """P1 盘点结论（已接受）：已有文件都不复用，每段 × 4 类全部补取。"""
    return tuple(
        RequestSpec(row.symbol, seg.ticker, cat, seg.start, seg.end)
        for row in identity
        for seg in row.segments
        for cat in CATEGORIES
    )
