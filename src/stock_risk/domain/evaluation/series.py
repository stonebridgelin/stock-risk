"""17 §2.1、§3 / D-03、D-05–D-07：公共价格序列、掩码与十进制 ATR。"""

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, localcontext
from enum import StrEnum

from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.evaluation.decimal_ratio import EVALUATION_CONTEXT, decimal_price
from stock_risk.domain.evaluation.params import ATR_PERIOD
from stock_risk.domain.lifecycle.history_segments import HistorySegment, split_segments
from stock_risk.domain.market_data import DailyBar
from stock_risk.domain.numeric import InvalidInputError
from stock_risk.domain.observation import DayPresence


class EndReason(StrEnum):
    DATA_END = "DATA_END"
    DELISTED = "DELISTED"
    REQUIRED_END = "REQUIRED_END"
    PERIOD_END = "PERIOD_END"


@dataclass(frozen=True, slots=True)
class EvalSeries:
    """calendar 仅含观察区间，bars 与其逐日对齐；缺价位置为 None。

    所有派生元组也与日历对齐（valid_days 除外）；valid_ordinals 在缺价日为 None。
    segment_ids 在短缺价期间保留历史段，断点当日起清空，直至下一条日线。
    """

    calendar: TradingCalendar
    bars: tuple[DailyBar | None, ...]
    gap_break: int
    mask_min: int
    end_reason: EndReason
    segments: tuple[HistorySegment, ...] = field(init=False)
    segment_ids: tuple[int | None, ...] = field(init=False)
    segment_counts: tuple[int, ...] = field(init=False)
    break_days: tuple[date, ...] = field(init=False)
    valid_days: tuple[date, ...] = field(init=False)
    valid_ordinals: tuple[int | None, ...] = field(init=False)
    mask: tuple[bool, ...] = field(init=False)
    closes: tuple[Decimal | None, ...] = field(init=False)
    atr20: tuple[Decimal | None, ...] = field(init=False)

    def __post_init__(self) -> None:
        _validate(self)
        presence = tuple(
            DayPresence(d, b is not None)
            for d, b in zip(self.calendar.days, self.bars, strict=True)
        )
        segments = split_segments(presence, self.gap_break)
        object.__setattr__(self, "segments", segments)
        with localcontext(EVALUATION_CONTEXT):
            _derive(self, segments)

    def index(self, day: date) -> int:
        i = bisect_left(self.calendar.days, day)
        if i == len(self.calendar.days) or self.calendar.days[i] != day:
            raise InvalidInputError(f"{day} 不在评价日历内")
        return i

    def close(self, day: date) -> Decimal:
        value = self.closes[self.index(day)]
        if value is None:
            raise InvalidInputError(f"{day} 缺价")
        return value


def _validate(series: EvalSeries) -> None:
    if len(series.bars) != len(series.calendar.days):
        raise InvalidInputError("评价日线必须与观察区间日历逐日对齐")
    for name in ("gap_break", "mask_min"):
        value = getattr(series, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise InvalidInputError(f"{name} 必须为正整数")
    if not isinstance(series.end_reason, EndReason):
        raise InvalidInputError("评价序列的终止原因不合法")
    for day, bar in zip(series.calendar.days, series.bars, strict=True):
        if bar is not None:
            if bar.day != day:
                raise InvalidInputError("评价日线的日期与日历不一致")
            op, hi, lo, cl = map(
                decimal_price, (bar.adj_open, bar.adj_high, bar.adj_low, bar.adj_close)
            )
            if hi < max(op, cl) or lo > min(op, cl):
                raise InvalidInputError(f"{day} 复权价高低点与开收盘不一致")


def _segment_atrs(series: EvalSeries, segments: tuple[HistorySegment, ...]) -> dict[date, Decimal]:
    """D-03：21 条有效日线才有 ATR20，TR 的前收跳过短缺价日；ATR 本身不量化。"""
    by_day = {b.day: b for b in series.bars if b is not None}
    result: dict[date, Decimal] = {}
    for segment in segments:
        trs: list[Decimal] = []
        for previous, day in zip(segment.bar_days, segment.bar_days[1:], strict=False):
            bar = by_day[day]
            hi, lo, prev = map(
                decimal_price, (bar.adj_high, bar.adj_low, by_day[previous].adj_close)
            )
            trs.append(max(hi - lo, abs(hi - prev), abs(lo - prev)))
            if len(trs) >= ATR_PERIOD:
                value = sum(trs[-ATR_PERIOD:], Decimal(0)) / Decimal(ATR_PERIOD)
                if value <= 0:
                    raise InvalidInputError(f"{day} 的评价 ATR20 非正")
                result[day] = value
    return result


def _derive(series: EvalSeries, segments: tuple[HistorySegment, ...]) -> None:
    segment_by_day = {
        day: (i, count)
        for i, seg in enumerate(segments)
        for count, day in enumerate(seg.bar_days, 1)
    }
    atrs = _segment_atrs(series, segments)
    ids, counts, breaks, ordinals, valid, closes, mask, atr20 = [], [], [], [], [], [], [], []
    missing, segment_id, count = 0, None, 0
    for day, bar in zip(series.calendar.days, series.bars, strict=True):
        if bar is None:
            missing += 1
            if missing == series.gap_break + 1:
                breaks.append(day)
                segment_id, count = None, 0
        else:
            missing = 0
            segment_id, count = segment_by_day[day]
        ids.append(segment_id)
        counts.append(count)
        ordinals.append(len(valid) if bar is not None else None)
        if bar is not None:
            valid.append(day)
        closes.append(decimal_price(bar.adj_close) if bar is not None else None)
        mask.append(bar is not None and count >= series.mask_min)
        atr20.append(atrs.get(day))
    for name, values in (
        ("segment_ids", ids),
        ("segment_counts", counts),
        ("break_days", breaks),
        ("valid_ordinals", ordinals),
        ("valid_days", valid),
        ("closes", closes),
        ("mask", mask),
        ("atr20", atr20),
    ):
        object.__setattr__(series, name, tuple(values))
