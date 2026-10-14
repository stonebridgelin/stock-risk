"""行情领域对象：日线与公司行动。纯数据，不含任何 I/O。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class DailyBar:
    """一根日线。原始价（不复权）与复权价同时保存，信号计算使用复权价。"""

    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_open: float
    adj_high: float
    adj_low: float
    adj_close: float


class ActionType(StrEnum):
    DIVIDEND = "dividend"
    SPLIT = "split"
    ADR_RATIO = "adr_ratio"


@dataclass(frozen=True, slots=True)
class CorporateAction:
    day: date
    type: ActionType
    value: float


@dataclass(frozen=True, slots=True)
class PriceHistory:
    """某个代码在一段区间内的日线与公司行动，按日期升序。"""

    bars: tuple[DailyBar, ...]
    actions: tuple[CorporateAction, ...]

    @property
    def first_day(self) -> date | None:
        return self.bars[0].day if self.bars else None

    @property
    def last_day(self) -> date | None:
        return self.bars[-1].day if self.bars else None


def quality_warnings(bars: tuple[DailyBar, ...]) -> list[str]:
    """只报告、不修改的数据质量检查。异常价格本身可能正是要测试的场景。"""
    warnings: list[str] = []
    for b in bars:
        if min(b.open, b.high, b.low, b.close) <= 0:
            warnings.append(f"{b.day}: 原始价出现非正值")
        if b.high < max(b.open, b.close) or b.low > min(b.open, b.close):
            warnings.append(f"{b.day}: 原始价高低点与开收盘不一致")
    return warnings
