"""指标（v0.4 第 2.1–2.3 节；I4、I18）。纯函数，输入为一个历史段内按日期升序的有效日线。

- 全部使用复权价（adj_*）；I11 登记的更正已在取数入口应用。
- I18：TR 中的 C_{t−1} 取同一历史段内上一条有效日线的收盘；ATR、SMA、窗口一律按有效日线计数，
  缺价日不占位。R2 例外（v0.4 R2-06），见 indicator_source.compute_r2。
- 下标 i 指日线在历史段中的位置（0 起）。历史不足时返回 None，由调用方映射为"不可计算"。
- 并列规则：窗口最高收盘并列取最近；区间最低收盘并列取最早。并列按量化后的数值精确比较。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from enum import StrEnum

from stock_risk.domain.market_data import DailyBar
from stock_risk.domain.numeric import InvalidInputError


class CalcStatus(StrEnum):
    OK = "OK"
    MISSING_DATA = "MISSING_DATA"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"


def _check_index(bars: Sequence[DailyBar], i: int) -> None:
    if not 0 <= i < len(bars):
        raise InvalidInputError(f"下标越界: {i}（共 {len(bars)} 条）")


def true_range(bars: Sequence[DailyBar], i: int) -> float:
    """TR_t = max(H − L, |H − C_prev|, |L − C_prev|)。

    C_prev 为同一历史段内上一条有效日线的收盘（I18），因此需要 i ≥ 1。
    """
    _check_index(bars, i)
    if i < 1:
        raise InvalidInputError("TR 需要上一条有效日线")
    b, prev_close = bars[i], bars[i - 1].adj_close
    return max(b.adj_high - b.adj_low, abs(b.adj_high - prev_close), abs(b.adj_low - prev_close))


def atr(bars: Sequence[DailyBar], i: int, n: int) -> float | None:
    """ATR_N(t) = fsum(TR_{t−N+1..t}) / N（简单平均）。

    需要 N + 1 条日线，不足返回 None；ATR ≤ 0 → InvalidInputError。
    """
    _check_index(bars, i)
    if n < 1:
        raise InvalidInputError(f"N 必须为正: {n}")
    if i < n:
        return None
    value = math.fsum(true_range(bars, j) for j in range(i - n + 1, i + 1)) / n
    if not value > 0:
        raise InvalidInputError(f"{bars[i].day}: ATR_{n} ≤ 0（{value}）")
    return value


def sma(bars: Sequence[DailyBar], i: int, m: int) -> float | None:
    """SMA_M(t) = fsum(最近 M 个收盘，含当日) / M；不足 M 条返回 None。"""
    _check_index(bars, i)
    if m < 1:
        raise InvalidInputError(f"M 必须为正: {m}")
    if i < m - 1:
        return None
    return math.fsum(b.adj_close for b in bars[i - m + 1 : i + 1]) / m


def window_max_close(
    bars: Sequence[DailyBar], i: int, w: int, since: date | None = None
) -> tuple[float, date]:
    """受限窗口：含 T 的最近 W 条有效日线。

    给定 since（last_release）时只取该日及之后；最高收盘并列取最近。
    """
    _check_index(bars, i)
    if w < 1:
        raise InvalidInputError(f"W 必须为正: {w}")
    window = [b for b in bars[max(0, i - w + 1) : i + 1] if since is None or b.day >= since]
    if not window:
        raise InvalidInputError(f"{bars[i].day}: 受限窗口为空")
    best = window[0]
    for b in window[1:]:
        if b.adj_close >= best.adj_close:
            best = b
    return best.adj_close, best.day


def min_close(bars: Sequence[DailyBar], start: date, i: int) -> tuple[float, date]:
    """[start, 第 i 条] 内的最低收盘及其日期，并列取最早。"""
    _check_index(bars, i)
    span = [b for b in bars[: i + 1] if b.day >= start]
    if not span:
        raise InvalidInputError(f"{start} ~ {bars[i].day} 内没有有效日线")
    best = span[0]
    for b in span[1:]:
        if b.adj_close < best.adj_close:
            best = b
    return best.adj_close, best.day


def max_close(bars: Sequence[DailyBar], start: date, end: date) -> float | None:
    """[start, end] 内有效日线的最高收盘；没有日线返回 None（BOX 区间最高用）。"""
    closes = [b.adj_close for b in bars if start <= b.day <= end]
    return max(closes) if closes else None


def max_close_prior(bars: Sequence[DailyBar], i: int, k: int) -> float | None:
    """今日之前最近 K 个有效交易日的最高收盘（不含当日，BOX_ROLL 用）；不足 K 条返回 None。"""
    _check_index(bars, i)
    if k < 1:
        raise InvalidInputError(f"K 必须为正: {k}")
    if i < k:
        return None
    return max(b.adj_close for b in bars[i - k : i])


def history_sufficient(valid_bars: int, w: int, n: int) -> bool:
    """当前历史段有效日线达到 W + N 条才可计算（v0.7 第 1.2 节第 3 步；13 号 S21a、S21b）。"""
    return valid_bars >= w + n
