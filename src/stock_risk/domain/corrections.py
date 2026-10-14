"""复权价更正（I11，T00a）：数据源复权价有已证实的错误时，读取时按登记的方法更正，原始文件不改。

每条更正必须有产品经理裁定编号与数据依据。目前只有一种方法：
- use_raw：区间内每条日线的 adj_open / adj_high / adj_low / adj_close 用对应的原始价替代。
  只适用于区间内没有拆股和分红的情况。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from enum import StrEnum

from stock_risk.domain.listing import ListingPeriod
from stock_risk.domain.market_data import DailyBar, PriceHistory


class CorrectionError(ValueError):
    """更正登记不合规，或不能应用到给定数据上。"""


class CorrectionMethod(StrEnum):
    USE_RAW = "use_raw"


def parse_method(text: str) -> CorrectionMethod:
    try:
        return CorrectionMethod(text)
    except ValueError as e:
        raise CorrectionError(f"未知的更正方法: {text!r}") from e


@dataclass(frozen=True, slots=True)
class PriceCorrection:
    """一条复权价更正，区间为闭区间 [start, end]。"""

    symbol: str
    start: date
    end: date
    method: CorrectionMethod
    ruling: str  # 产品经理裁定编号，如 PM-03
    evidence: str  # 数据依据

    def __post_init__(self) -> None:
        if not self.symbol:
            raise CorrectionError("更正的 symbol 不能为空")
        if self.start > self.end:
            raise CorrectionError(f"{self.symbol}: 更正起始日晚于结束日")
        if not isinstance(self.method, CorrectionMethod):
            raise CorrectionError(f"{self.symbol}: 未知的更正方法 {self.method!r}")
        if not self.ruling or not self.evidence:
            raise CorrectionError(f"{self.symbol}: 更正必须有裁定编号与数据依据")

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end


def validate_correction(
    correction: PriceCorrection, history: PriceHistory, listing: ListingPeriod | None
) -> None:
    """更正区间必须落在上市期间内；use_raw 的区间内不得有公司行动。"""
    c = correction
    if listing is not None and not (listing.contains(c.start) and listing.contains(c.end)):
        raise CorrectionError(f"{c.symbol}: 更正区间 {c.start} ~ {c.end} 超出上市期间")
    if c.method is CorrectionMethod.USE_RAW:
        inside = [a for a in history.actions if c.contains(a.day)]
        if inside:
            days = ", ".join(f"{a.day} {a.type}" for a in inside)
            raise CorrectionError(f"{c.symbol}: use_raw 区间内有公司行动（{days}）")


def _use_raw(b: DailyBar) -> DailyBar:
    return replace(b, adj_open=b.open, adj_high=b.high, adj_low=b.low, adj_close=b.close)


def apply_corrections(
    history: PriceHistory,
    corrections: Sequence[PriceCorrection],
    listing: ListingPeriod | None = None,
) -> PriceHistory:
    """先逐条校验，再按登记顺序应用。区间外的日线与全部公司行动保持不变。"""
    for c in corrections:
        validate_correction(c, history, listing)
    bars = list(history.bars)
    for c in corrections:
        if c.method is CorrectionMethod.USE_RAW:
            bars = [_use_raw(b) if c.contains(b.day) else b for b in bars]
        else:  # pragma: no cover - 新增方法时必须在此实现
            raise CorrectionError(f"更正方法未实现: {c.method}")
    return PriceHistory(tuple(bars), history.actions)
