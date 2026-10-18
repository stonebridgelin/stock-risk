"""原始数据的文本行与解析（I9、I12 第 2 步）。

基础设施层只把 CSV 读成文本行（日期已解析，数值保持文本）；价格从文本直接量化，
不经过 float 再转换。解析时执行 v0.4 第 2.0 节的"明确失败"检查，原始价与复权价分别检查。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from stock_risk.domain.market_data import ActionType, CorporateAction, DailyBar, PriceHistory
from stock_risk.domain.numeric import InvalidInputError, parse_finite, quantize_price


@dataclass(frozen=True, slots=True)
class RawBarText:
    day: date
    open: str
    high: str
    low: str
    close: str
    volume: str
    adj_open: str
    adj_high: str
    adj_low: str
    adj_close: str
    # 缺列或空单元格为 None（未知）；"0" 是真实的零（T11-D0a 第 2 节）
    adj_volume: str | None = None


@dataclass(frozen=True, slots=True)
class RawActionText:
    day: date
    type: str
    value: str


def _check_ohlc(
    symbol: str, day: date, basis: str, o: float, h: float, lo: float, c: float
) -> None:
    if h < lo:
        raise InvalidInputError(f"{symbol} {day} {basis}: 高 < 低（H={h} L={lo}）")
    if not lo <= c <= h:
        raise InvalidInputError(f"{symbol} {day} {basis}: 收盘不在 [低, 高]（C={c} L={lo} H={h}）")


def parse_adj_volume(text: str | None, field: str) -> float | None:
    """adj_volume 的读法（T11-D0a 第 2.2 节）：None 或空 → None（未知）；"0" → 0.0（真实的零）；
    负数、非有限、无法解析 → InvalidInputError，不当作缺失。"""
    if text is None or not text.strip():
        return None
    value = parse_finite(text, field)
    if value < 0:
        raise InvalidInputError(f"{field} 为负数: {text!r}")
    return value


def parse_bar(symbol: str, row: RawBarText) -> DailyBar:
    def q(text: str, name: str) -> float:
        return quantize_price(text, f"{symbol} {row.day} {name}")

    bar = DailyBar(
        day=row.day,
        open=q(row.open, "open"),
        high=q(row.high, "high"),
        low=q(row.low, "low"),
        close=q(row.close, "close"),
        volume=parse_finite(row.volume, f"{symbol} {row.day} volume"),
        adj_open=q(row.adj_open, "adj_open"),
        adj_high=q(row.adj_high, "adj_high"),
        adj_low=q(row.adj_low, "adj_low"),
        adj_close=q(row.adj_close, "adj_close"),
        adj_volume=parse_adj_volume(row.adj_volume, f"{symbol} {row.day} adj_volume"),
    )
    _check_ohlc(symbol, row.day, "原始价", bar.open, bar.high, bar.low, bar.close)
    _check_ohlc(symbol, row.day, "复权价", bar.adj_open, bar.adj_high, bar.adj_low, bar.adj_close)
    return bar


def parse_action(symbol: str, row: RawActionText) -> CorporateAction:
    try:
        kind = ActionType(row.type)
    except ValueError as e:
        raise InvalidInputError(f"{symbol} {row.day}: 未知公司行动类型 {row.type!r}") from e
    return CorporateAction(row.day, kind, parse_finite(row.value, f"{symbol} {row.day} {kind}"))


def parse_history(
    symbol: str, bars: tuple[RawBarText, ...], actions: tuple[RawActionText, ...]
) -> PriceHistory:
    """逐行解析；日期必须严格升序。"""
    parsed = tuple(parse_bar(symbol, r) for r in bars)
    days = [b.day for b in parsed]
    if any(a >= b for a, b in zip(days, days[1:], strict=False)):
        raise InvalidInputError(f"{symbol}: 日线日期未严格升序（重复或乱序）")
    acts = tuple(sorted((parse_action(symbol, r) for r in actions), key=lambda a: (a.day, a.type)))
    return PriceHistory(parsed, acts)
