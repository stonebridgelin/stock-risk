"""指标接口（I5）：生命周期引擎取数用。两种实现：

- ComputedIndicators：由一个历史段内的日线计算（回放与 F 组）；
- TableIndicators：开盘、收盘按表给定，ATR、SMA 按表给定（S 组"给定 ATR、给定 SMA10"）。

窗口类量（窗口最高、区间最低、区间最高、前 K 日最高、有效日计数）在两种实现中都由收盘序列计算，
规则相同。R2 跳空由 compute_r2 统一按接口计算，不让实现自行推测前一交易日收盘（13 号 S19）。
一个实例只代表一个历史段；按 G 切分历史段属于生命周期（T02）。
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from stock_risk.domain import indicators as ind
from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.indicators import CalcStatus
from stock_risk.domain.market_data import DailyBar
from stock_risk.domain.numeric import (
    DEFAULT_TOLERANCES,
    InvalidInputError,
    Kind,
    Tolerances,
    le,
    lt,
)


class IndicatorSource(Protocol):
    """某个历史段的取数接口。day 必须是有日线的交易日，否则抛 InvalidInputError（has_bar 除外）。"""

    @property
    def calendar(self) -> TradingCalendar: ...

    def has_bar(self, day: date) -> bool: ...

    def open(self, day: date) -> float: ...

    def close(self, day: date) -> float: ...

    def valid_bar_count(self, day: date) -> int:
        """当前历史段内截至 day（含）的有效日线数，用于 W + N 判定。"""
        ...

    def count_valid_after(self, start: date, day: date) -> int:
        """start 之后（不含）到 day（含）的有效日线数（since_obs）。"""
        ...

    def atr(self, day: date, n: int) -> float | None:
        """ATR_N(day)；历史不足为 None。"""
        ...

    def sma(self, day: date, m: int) -> float | None: ...

    def window_max_close(self, day: date, w: int, since: date | None) -> tuple[float, date]:
        """受限窗口最高收盘与峰值日（并列取最近）。"""
        ...

    def min_close(self, start: date, day: date) -> tuple[float, date]:
        """[start, day] 最低收盘及日期（并列取最早）。"""
        ...

    def max_close(self, start: date, end: date) -> float | None:
        """[start, end] 最高收盘（BOX 区间最高）。"""
        ...

    def max_close_prior(self, day: date, k: int) -> float | None:
        """day 之前最近 K 个有效交易日的最高收盘（BOX_ROLL）。"""
        ...


@dataclass(frozen=True, slots=True)
class R2Result:
    status: CalcStatus  # OK / MISSING_DATA（前一交易日无日线）/ INSUFFICIENT_HISTORY
    gap_score: float | None  # 仅 OK 时有值
    triggered: bool


def compute_r2(
    src: IndicatorSource, day: date, n: int, g: float, tol: Tolerances = DEFAULT_TOLERANCES
) -> R2Result:
    """G_t = (O_t − C_prev) / ATR_N(prev)；prev 必须是交易日历上 T 的前一个交易日且有日线（R2-06）。

    触发：状态 OK 且 G_t ≤ −g（比率 ε）且 C_T < C_prev（价格 ε）。
    """
    prev = src.calendar.previous(day)
    if prev is None or not src.has_bar(prev):
        return R2Result(CalcStatus.MISSING_DATA, None, False)
    prev_atr = src.atr(prev, n)
    if prev_atr is None:
        return R2Result(CalcStatus.INSUFFICIENT_HISTORY, None, False)
    prev_close = src.close(prev)
    score = (src.open(day) - prev_close) / prev_atr
    triggered = le(score, -g, Kind.RATIO, tol) and lt(src.close(day), prev_close, Kind.PRICE, tol)
    return R2Result(CalcStatus.OK, score, triggered)


class _ClosesBase:
    """两种实现共用：按升序日期保存收盘，计算窗口类量。"""

    def __init__(self, calendar: TradingCalendar, days: Sequence[date]) -> None:
        if any(a >= b for a, b in zip(days, days[1:], strict=False)):
            raise InvalidInputError("日线日期未严格升序")
        stray = [d for d in days if not calendar.contains(d)]
        if stray:
            raise InvalidInputError(f"日线日期不在交易日历中: {stray[:5]}")
        self._calendar = calendar
        self._days = tuple(days)
        self._pos = {d: i for i, d in enumerate(self._days)}

    @property
    def calendar(self) -> TradingCalendar:
        return self._calendar

    def has_bar(self, day: date) -> bool:
        return day in self._pos

    def _i(self, day: date) -> int:
        try:
            return self._pos[day]
        except KeyError:
            raise InvalidInputError(f"{day} 没有日线") from None

    def count_valid_after(self, start: date, day: date) -> int:
        lo = bisect.bisect_right(self._days, start)
        hi = bisect.bisect_right(self._days, day)
        return max(0, hi - lo)


class ComputedIndicators(_ClosesBase):
    """由日线计算。bars 为一个历史段内的全部有效日线（复权价，已量化与更正）。"""

    def __init__(self, calendar: TradingCalendar, bars: Sequence[DailyBar]) -> None:
        super().__init__(calendar, [b.day for b in bars])
        self._bars = tuple(bars)

    def open(self, day: date) -> float:
        return self._bars[self._i(day)].adj_open

    def close(self, day: date) -> float:
        return self._bars[self._i(day)].adj_close

    def valid_bar_count(self, day: date) -> int:
        return bisect.bisect_right(self._days, day)

    def atr(self, day: date, n: int) -> float | None:
        return ind.atr(self._bars, self._i(day), n)

    def sma(self, day: date, m: int) -> float | None:
        return ind.sma(self._bars, self._i(day), m)

    def window_max_close(self, day: date, w: int, since: date | None) -> tuple[float, date]:
        return ind.window_max_close(self._bars, self._i(day), w, since)

    def min_close(self, start: date, day: date) -> tuple[float, date]:
        return ind.min_close(self._bars, start, self._i(day))

    def max_close(self, start: date, end: date) -> float | None:
        return ind.max_close(self._bars, start, end)

    def max_close_prior(self, day: date, k: int) -> float | None:
        return ind.max_close_prior(self._bars, self._i(day), k)


@dataclass(frozen=True, slots=True)
class TableRow:
    """一个有日线的交易日：开盘、收盘按表给定；atr、sma 为表中的给定值（没有给定则为 None）。"""

    open: float
    close: float
    atr: float | None = None
    sma: float | None = None


class TableIndicators(_ClosesBase):
    """按表给定（S 组）。给定的 ATR 只对声明的 n 有效，给定的 SMA 只对声明的 sma_m 有效。

    prior_valid_bars：表首日之前已有的有效日线数（"历史长度视为充足"时给一个足够大的值）。
    """

    def __init__(
        self,
        calendar: TradingCalendar,
        rows: Mapping[date, TableRow],
        n: int,
        sma_m: int = 10,
        prior_valid_bars: int = 0,
    ) -> None:
        days = sorted(rows)
        super().__init__(calendar, days)
        self._rows = dict(rows)
        self._n = n
        self._sma_m = sma_m
        self._prior = prior_valid_bars

    def _row(self, day: date) -> TableRow:
        self._i(day)
        return self._rows[day]

    def open(self, day: date) -> float:
        return self._row(day).open

    def close(self, day: date) -> float:
        return self._row(day).close

    def valid_bar_count(self, day: date) -> int:
        return self._prior + bisect.bisect_right(self._days, day)

    def atr(self, day: date, n: int) -> float | None:
        if n != self._n:
            raise InvalidInputError(f"表中给定的是 ATR_{self._n}，不能取 ATR_{n}")
        return self._row(day).atr

    def sma(self, day: date, m: int) -> float | None:
        if m != self._sma_m:
            raise InvalidInputError(f"表中给定的是 SMA{self._sma_m}，不能取 SMA{m}")
        return self._row(day).sma

    def _closes(self) -> list[DailyBar]:
        # 窗口类量只用收盘；借用 DailyBar 以复用 indicators 中的同一套规则
        return [
            DailyBar(d, r.open, r.close, r.close, r.close, 0, r.open, r.close, r.close, r.close)
            for d, r in ((d, self._rows[d]) for d in self._days)
        ]

    def window_max_close(self, day: date, w: int, since: date | None) -> tuple[float, date]:
        return ind.window_max_close(self._closes(), self._i(day), w, since)

    def min_close(self, start: date, day: date) -> tuple[float, date]:
        return ind.min_close(self._closes(), start, self._i(day))

    def max_close(self, start: date, end: date) -> float | None:
        return ind.max_close(self._closes(), start, end)

    def max_close_prior(self, day: date, k: int) -> float | None:
        return ind.max_close_prior(self._closes(), self._i(day), k)
