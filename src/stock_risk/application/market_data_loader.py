"""读取原始数据供下游使用的统一路径：读取 → 应用已登记的复权价更正（I11）。

回放等下游一律通过这里取数，不直接用 RawDataReader，保证更正处处一致。
审计需要同时看未更正与更正后的数据，因此两者都返回。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from stock_risk.application.ports import (
    ListingPeriodProvider,
    PriceCorrectionProvider,
    RawDataReader,
)
from stock_risk.domain.corrections import PriceCorrection, apply_corrections
from stock_risk.domain.listing import ListingPeriod
from stock_risk.domain.market_data import PriceHistory


@dataclass(frozen=True, slots=True)
class LoadedHistory:
    symbol: str
    raw: PriceHistory  # 数据源原样
    corrected: PriceHistory  # 应用更正后；没有更正时与 raw 相同
    corrections: tuple[PriceCorrection, ...]
    listing: ListingPeriod | None
    sealed_rows: int


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

    def load(self, symbol: str) -> LoadedHistory:
        """更正不合规（超出上市期间、区间内有公司行动等）时抛 CorrectionError，不静默跳过。"""
        raw = self._reader.load(symbol)
        listing = self._listings.get(symbol)
        corrs = self._corrections.get(symbol, ())
        corrected = apply_corrections(raw.history, corrs, listing) if corrs else raw.history
        return LoadedHistory(symbol, raw.history, corrected, corrs, listing, raw.sealed_rows)
