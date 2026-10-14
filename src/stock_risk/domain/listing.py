"""上市期间（T00 / D2、D3）：证券在产品范围内交易所（NYSE / Nasdaq）挂牌的区间。

原始数据不截断；上市期间只在审计和回放中使用。日期为空表示该端不设边界
（未登记，或找不到可靠来源而留空，后者需在配置备注与审计报告中说明）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from stock_risk.domain.market_data import DailyBar, PriceHistory


@dataclass(frozen=True, slots=True)
class ListingPeriod:
    symbol: str
    listed_from: date | None = None
    listed_until: date | None = None
    venue: str = ""
    source_url: str = ""
    verified_on: date | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("上市期间的 symbol 不能为空")
        if (
            self.listed_from is not None
            and self.listed_until is not None
            and self.listed_from > self.listed_until
        ):
            raise ValueError(f"{self.symbol}: 上市起始日晚于退市日")
        if self.is_bounded and not self.source_url:
            raise ValueError(f"{self.symbol}: 登记了上市期间日期但没有来源链接")

    @property
    def is_bounded(self) -> bool:
        return self.listed_from is not None or self.listed_until is not None

    def contains(self, day: date) -> bool:
        return (self.listed_from is None or day >= self.listed_from) and (
            self.listed_until is None or day <= self.listed_until
        )


def clip_bars_to_listing(
    bars: Sequence[DailyBar], period: ListingPeriod | None
) -> tuple[DailyBar, ...]:
    """按上市期间截取日线；没有上市期间配置时原样返回。"""
    if period is None:
        return tuple(bars)
    return tuple(b for b in bars if period.contains(b.day))


def clip_history_to_listing(history: PriceHistory, period: ListingPeriod | None) -> PriceHistory:
    """按上市期间截取日线与公司行动。"""
    if period is None:
        return history
    return PriceHistory(
        bars=clip_bars_to_listing(history.bars, period),
        actions=tuple(a for a in history.actions if period.contains(a.day)),
    )
