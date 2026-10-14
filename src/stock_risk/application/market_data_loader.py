"""统一取数入口（I12）：下游（回放、图表、汇总、以后的评价）一律经这里取数。

load 的处理顺序固定为：
  ① 封存检查：文件（日线与公司行动）中任何日期晚于 2019-12-31 → 立即抛 SealViolation，
     不解析该行价格，也不先过滤后继续；
  ② 解析与量化（I9）：价格从 CSV 文本直接量化到六位小数；
  ③ 复权更正（I11）；
  ④ 按观察区间截取（I2、I3、I10）；
  ⑤ 第二次封存检查。
另给出三个区间与观察区间内逐日"有日线 / 缺价"序列。

审计走 audit_view：越界行只计数、不报错，数值不量化。两条路径分开，审计不调用 load。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from stock_risk.application.ports import (
    ListingPeriodProvider,
    PriceCorrectionProvider,
    RawDataReader,
)
from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.corrections import PriceCorrection, apply_corrections
from stock_risk.domain.listing import ListingPeriod
from stock_risk.domain.market_data import PriceHistory
from stock_risk.domain.numeric import InvalidInputError
from stock_risk.domain.observation import (
    DayPresence,
    SymbolRanges,
    compute_ranges,
    daily_presence,
)
from stock_risk.domain.raw_text import parse_history
from stock_risk.domain.seal import assert_no_sealed_days, assert_sealed

# I1：交易日历代理
CALENDAR_SYMBOL = "SPY"
# 探索期原始数据的下载请求起点（fetch-raw 默认值）；其后首个交易日用于判定 request_start
DOWNLOAD_START = date(2008, 1, 1)


@dataclass(frozen=True, slots=True)
class LoadedHistory:
    """审计视图：数据源原样与更正后（均不量化），越界行只计数。"""

    symbol: str
    raw: PriceHistory
    corrected: PriceHistory
    corrections: tuple[PriceCorrection, ...]
    listing: ListingPeriod | None
    sealed_rows: int


@dataclass(frozen=True, slots=True)
class ObservedData:
    """供下游使用：已量化、已更正、已截取到观察区间。"""

    symbol: str
    history: PriceHistory  # 观察区间内的日线与公司行动
    ranges: SymbolRanges
    presence: tuple[DayPresence, ...]  # 观察区间内逐个交易日"有日线 / 缺价"
    corrections: tuple[PriceCorrection, ...]
    listing: ListingPeriod | None

    @property
    def missing_days(self) -> tuple[date, ...]:
        return tuple(p.day for p in self.presence if not p.has_bar)


class MarketDataLoader:
    def __init__(
        self,
        reader: RawDataReader,
        listings: ListingPeriodProvider,
        corrections: PriceCorrectionProvider,
    ) -> None:
        self._reader = reader
        self._listings: Mapping[str, ListingPeriod] = listings.load()
        self._corrections: Mapping[str, tuple[PriceCorrection, ...]] = corrections.load()
        self._calendar: TradingCalendar | None = None

    @property
    def reader(self) -> RawDataReader:
        return self._reader

    @property
    def listings(self) -> Mapping[str, ListingPeriod]:
        return self._listings

    @property
    def corrections(self) -> Mapping[str, tuple[PriceCorrection, ...]]:
        return self._corrections

    def exists(self, symbol: str) -> bool:
        return self._reader.exists(symbol)

    # ---- 下游路径（I12）----
    def _parsed(self, symbol: str) -> PriceHistory:
        """第 ①、② 步。"""
        text = self._reader.load_text(symbol)
        assert_no_sealed_days((r.day for r in text.bars), f"{symbol}.csv")
        assert_no_sealed_days((r.day for r in text.actions), f"{symbol}.actions.csv")
        return parse_history(symbol, text.bars, text.actions)

    def calendar(self) -> TradingCalendar:
        """交易日历代理：SPY 文件全部日线的日期（经第 ①、② 步）。"""
        if self._calendar is None:
            self._calendar = TradingCalendar.from_days(
                b.day for b in self._parsed(CALENDAR_SYMBOL).bars
            )
        return self._calendar

    def load(self, symbol: str) -> ObservedData:
        history = self._parsed(symbol)
        if not history.bars:
            raise InvalidInputError(f"{symbol}: 没有日线")
        listing = self._listings.get(symbol)
        corrs = self._corrections.get(symbol, ())
        if corrs:  # ③
            history = apply_corrections(history, corrs, listing)
        calendar = self.calendar()
        request_first = calendar.first_on_or_after(DOWNLOAD_START)
        if request_first is None:
            raise InvalidInputError("交易日历中没有下载起点之后的交易日")
        ranges = compute_ranges(
            symbol, history.bars[0].day, history.bars[-1].day, listing, calendar, request_first
        )
        lo, hi = ranges.observation_start, ranges.observation_end  # ④
        observed = PriceHistory(
            tuple(b for b in history.bars if lo <= b.day <= hi),
            tuple(a for a in history.actions if lo <= a.day <= hi),
        )
        assert_sealed(observed)  # ⑤
        presence = daily_presence(calendar, [b.day for b in observed.bars], ranges)
        return ObservedData(symbol, observed, ranges, presence, corrs, listing)

    # ---- 审计路径 ----
    def audit_view(self, symbol: str) -> LoadedHistory:
        """更正不合规（超出上市期间、区间内有公司行动等）时抛 CorrectionError，不静默跳过。"""
        raw = self._reader.load(symbol)
        listing = self._listings.get(symbol)
        corrs = self._corrections.get(symbol, ())
        corrected = apply_corrections(raw.history, corrs, listing) if corrs else raw.history
        return LoadedHistory(symbol, raw.history, corrected, corrs, listing, raw.sealed_rows)
