"""Massive 付费档验证（T08d-2b 第 1、2 节）：日线、拆股与分红的封存处理、统计与对照。纯函数。

封存：
- 日线请求的 to 不得晚于 2019-12-31（check_price_range）；响应中晚于截止日的行丢弃并计数，不落盘；
- 拆股执行日、分红除息日晚于截止日的行丢弃并计数；分红的其他日期字段（宣布日、登记日、派息日）
  晚于截止日的写为空，记入 masked_after_2019，不保留真实日期；
- 不保存 historical_adjustment_factor、split_adjusted_cash_amount（二者按截止日之后的事件累计）。
照实统计，不作结论。
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from stock_risk.domain.massive_followup import FOLLOWUP_CASES
from stock_risk.domain.massive_probe import CASES
from stock_risk.domain.seal import EXPLORATION_END, SealViolation, assert_dates_not_sealed

PRICE_FROM = date(2008, 1, 2)
PRICE_TO = EXPLORATION_END
COMPARE_TICKERS = ("MSFT", "XOM", "KO")  # 与 Tiingo 对照
REUSE_TICKERS = ("GM", "DELL", "ADT")  # 代码复用
SEGMENT_GAP_DAYS = (
    20  # 报告用：缺口不少于 20 个交易日时视为两段价格（产品经理未给值，回执中待确认）
)
EXAMPLES = 10
CLOSE_TOL = 0.01  # 第 1 节：收盘价差超过 0.01 的天数
ADJ_TOL = 1e-6  # 第 2 节：自算拆股调整与 adjusted=true 的相对差阈值
AMOUNT_TOL = 1e-6  # 分红金额、拆股比例视为一致的绝对差


def paid_tickers() -> tuple[str, ...]:
    """第 1 节对象：T08d-2a 第 2.2 节 21 个案例、第 2.4 节 C 的 6 个案例、MSFT/XOM/KO。

    同一代码只取一次。
    """
    out: list[str] = []
    for t in [*(c.ticker for c in CASES), *(c.ticker for c in FOLLOWUP_CASES), *COMPARE_TICKERS]:
        if t not in out:
            out.append(t)
    return tuple(out)


def check_price_range(start: date, end: date) -> None:
    if end > EXPLORATION_END:
        raise SealViolation(f"Massive 日线请求 to={end} 晚于 {EXPLORATION_END}，拒绝执行")
    if start > end:
        raise ValueError("from 晚于 to")


def _day(raw: Any) -> date | None:
    if raw is None or raw == "":
        return None
    return date.fromisoformat(str(raw)[:10])


# ---- 日线 ----


@dataclass(frozen=True, slots=True)
class Bar:
    """日线一行。只保留日期、开高低收、成交量、VWAP、成交笔数。"""

    day: date
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None
    vwap: float | None
    trades: int | None

    def __post_init__(self) -> None:
        assert_dates_not_sealed("Massive 日线", self.day)


def _num(v: Any) -> float | None:
    return None if v is None or v == "" else float(v)


def bar_day(t_ms: Any) -> date:
    """t 为窗口起点的 Unix 毫秒时间戳（美东零点，即 UTC 04:00 或 05:00），UTC 日期即交易日。"""
    return datetime.fromtimestamp(int(t_ms) / 1000, tz=UTC).date()


def seal_bars(raws: Iterable[Mapping[str, Any]]) -> tuple[list[Bar], int]:
    """响应行 → Bar（按日期升序）。晚于截止日的行丢弃，返回丢弃条数。"""
    bars: list[Bar] = []
    dropped = 0
    for r in raws:
        d = bar_day(r["t"])
        if d > EXPLORATION_END:
            dropped += 1
            continue
        n = r.get("n")
        bars.append(
            Bar(
                d,
                _num(r.get("o")),
                _num(r.get("h")),
                _num(r.get("l")),
                _num(r.get("c")),
                _num(r.get("v")),
                _num(r.get("vw")),
                None if n is None else int(n),
            )
        )
    bars.sort(key=lambda b: b.day)
    return bars, dropped


def missing_runs(days: Sequence[date], calendar: Sequence[date]) -> list[tuple[date, date, int]]:
    """首日到末日之间，交易日历中有、日线中没有的连续缺口：(起, 止, 交易日数)。"""
    if not days:
        return []
    have = set(days)
    span = [d for d in calendar if days[0] <= d <= days[-1]]
    runs: list[tuple[date, date, int]] = []
    start: date | None = None
    prev: date | None = None
    n = 0
    for d in span:
        if d in have:
            if start is not None and prev is not None:
                runs.append((start, prev, n))
            start, n = None, 0
        else:
            start = start or d
            n += 1
        prev = d
    return runs


def bar_stats(bars: Sequence[Bar], calendar: Sequence[date]) -> dict[str, Any]:
    """首日、末日、行数；按交易日历的缺失天数与最长缺口；成交量为 0 的天数；日历外的日期数。"""
    if not bars:
        return {"rows": 0}
    days = [b.day for b in bars]
    runs = missing_runs(days, calendar)
    longest = max(runs, key=lambda r: (r[2], -r[0].toordinal()), default=None)  # 同长取最早
    cal = set(calendar)
    return {
        "first": days[0].isoformat(),
        "last": days[-1].isoformat(),
        "rows": len(bars),
        "missing_days": sum(r[2] for r in runs),
        "longest_gap": None
        if longest is None
        else {"from": longest[0].isoformat(), "to": longest[1].isoformat(), "days": longest[2]},
        "zero_volume_days": sum(1 for b in bars if b.volume == 0),
        "off_calendar_days": sum(1 for d in days if d not in cal),
    }


def segments(
    bars: Sequence[Bar], calendar: Sequence[date], min_gap: int = SEGMENT_GAP_DAYS
) -> list[dict[str, Any]]:
    """代码复用：缺口不少于 min_gap 个交易日处分段，列出每段起止与到下一段的空档。"""
    if not bars:
        return []
    days = [b.day for b in bars]
    cuts = [r for r in missing_runs(days, calendar) if r[2] >= min_gap]
    out: list[dict[str, Any]] = []
    seg_start = days[0]
    for gap_from, gap_to, n in cuts:
        seg_end = max(d for d in days if d < gap_from)
        out.append(
            {
                "from": seg_start.isoformat(),
                "to": seg_end.isoformat(),
                "rows": sum(1 for d in days if seg_start <= d <= seg_end),
                "gap_to_next_days": n,
            }
        )
        seg_start = min(d for d in days if d > gap_to)
    out.append(
        {
            "from": seg_start.isoformat(),
            "to": days[-1].isoformat(),
            "rows": sum(1 for d in days if d >= seg_start),
            "gap_to_next_days": None,
        }
    )
    return out


@dataclass(frozen=True, slots=True)
class ReferenceBar:
    """Tiingo 原始价（data/raw_selection/）一行：未复权收盘价与成交量。"""

    day: date
    close: float
    volume: float

    def __post_init__(self) -> None:
        assert_dates_not_sealed("Tiingo 日线", self.day)


def compare_prices(bars: Sequence[Bar], ref: Sequence[ReferenceBar]) -> dict[str, Any]:
    """与 Tiingo 逐日比较：日期集合差异、收盘价差最大值与超过 0.01 的天数、成交量相对差中位数。"""
    m = {b.day: b for b in bars}
    t = {r.day: r for r in ref}
    only_m, only_t = sorted(set(m) - set(t)), sorted(set(t) - set(m))
    common = sorted(set(m) & set(t))
    diffs = [abs((m[d].close or 0.0) - t[d].close) for d in common if m[d].close is not None]
    vol = [
        abs((m[d].volume or 0.0) - t[d].volume) / t[d].volume
        for d in common
        if t[d].volume > 0 and m[d].volume is not None
    ]
    worst = max(common, key=lambda d: abs((m[d].close or 0.0) - t[d].close), default=None)
    return {
        "common_days": len(common),
        "only_massive": len(only_m),
        "only_massive_examples": [d.isoformat() for d in only_m[:EXAMPLES]],
        "only_tiingo": len(only_t),
        "only_tiingo_examples": [d.isoformat() for d in only_t[:EXAMPLES]],
        "close_abs_diff_max": round(max(diffs), 6) if diffs else None,
        "close_abs_diff_max_day": worst.isoformat() if worst else None,
        "close_diff_over_0_01_days": sum(1 for x in diffs if x > CLOSE_TOL),
        "volume_rel_diff_median": round(statistics.median(vol), 6) if vol else None,
    }


# ---- 拆股与分红 ----


@dataclass(frozen=True, slots=True)
class Split:
    day: date  # execution_date
    split_from: float
    split_to: float
    adjustment_type: str | None

    def __post_init__(self) -> None:
        assert_dates_not_sealed("Massive 拆股", self.day)

    @property
    def factor(self) -> float:
        """新股数 / 旧股数（与 Tiingo 的 splitFactor 同向）。"""
        return self.split_to / self.split_from


def seal_split(raw: Mapping[str, Any]) -> Split | None:
    """执行日晚于截止日的返回 None（丢弃，由调用方计数）。"""
    d = _day(raw.get("execution_date"))
    if d is None or d > EXPLORATION_END:
        return None
    at = raw.get("adjustment_type")
    return Split(
        d, float(raw["split_from"]), float(raw["split_to"]), None if at is None else str(at)
    )


DIVIDEND_OTHER_DATES = ("declaration_date", "record_date", "pay_date")


@dataclass(frozen=True, slots=True)
class Dividend:
    day: date  # ex_dividend_date
    cash_amount: float | None
    currency: str | None
    declaration_date: date | None
    record_date: date | None
    pay_date: date | None
    frequency: int | None
    distribution_type: str | None
    masked_after_2019: tuple[str, ...]  # 晚于截止日、已写为空的日期字段名

    def __post_init__(self) -> None:
        assert_dates_not_sealed(
            "Massive 分红", self.day, self.declaration_date, self.record_date, self.pay_date
        )


def seal_dividend(raw: Mapping[str, Any]) -> Dividend | None:
    """除息日晚于截止日的返回 None（丢弃）；其他日期字段晚于截止日的写为空并记录字段名。"""
    d = _day(raw.get("ex_dividend_date"))
    if d is None or d > EXPLORATION_END:
        return None
    other: dict[str, date | None] = {}
    masked: list[str] = []
    for f in DIVIDEND_OTHER_DATES:
        v = _day(raw.get(f))
        if v is not None and v > EXPLORATION_END:
            masked.append(f)
            v = None
        other[f] = v
    freq = raw.get("frequency")
    cur, dt = raw.get("currency"), raw.get("distribution_type")
    return Dividend(
        d,
        _num(raw.get("cash_amount")),
        None if cur is None else str(cur),
        other["declaration_date"],
        other["record_date"],
        other["pay_date"],
        None if freq is None else int(freq),
        None if dt is None else str(dt),
        tuple(masked),
    )


@dataclass(frozen=True, slots=True)
class ReferenceAction:
    """Tiingo 公司行动（data/raw_selection/<代码>.actions.csv）：dividend 金额或 split 比例。"""

    day: date
    type: str  # dividend | split
    value: float


def _match(mine: Mapping[date, list[float]], ref: Mapping[date, list[float]]) -> dict[str, Any]:
    both_ok, value_diff, only_m, only_t = [], [], [], []
    for d in sorted(set(mine) | set(ref)):
        a, b = mine.get(d, []), ref.get(d, [])
        if a and b:
            same = len(a) == len(b) and all(
                abs(x - y) <= AMOUNT_TOL for x, y in zip(sorted(a), sorted(b), strict=True)
            )
            if same:
                both_ok.append(d)
            else:
                value_diff.append((d, a, b))
        elif a:
            only_m.append((d, a))
        else:
            only_t.append((d, b))
    return {
        "match": len(both_ok),
        "same_date_value_differs": len(value_diff),
        "same_date_value_differs_examples": [
            {"date": d.isoformat(), "massive": a, "tiingo": b} for d, a, b in value_diff[:EXAMPLES]
        ],
        "only_massive": len(only_m),
        "only_massive_examples": [
            {"date": d.isoformat(), "massive": a} for d, a in only_m[:EXAMPLES]
        ],
        "only_tiingo": len(only_t),
        "only_tiingo_examples": [
            {"date": d.isoformat(), "tiingo": b} for d, b in only_t[:EXAMPLES]
        ],
    }


def compare_actions(
    splits: Sequence[Split],
    dividends: Sequence[Dividend],
    ref: Sequence[ReferenceAction],
    start: date = PRICE_FROM,
    end: date = PRICE_TO,
) -> dict[str, Any]:
    """与 Tiingo 逐条比较（只比较 Tiingo 覆盖的区间）。

    分为日期与金额（比例）一致、同日不一致、只在一边。
    """

    def group(items: Iterable[tuple[date, float]]) -> dict[date, list[float]]:
        out: dict[date, list[float]] = {}
        for d, v in items:
            if start <= d <= end:
                out.setdefault(d, []).append(v)
        return out

    return {
        "dividends": _match(
            group((x.day, x.cash_amount) for x in dividends if x.cash_amount is not None),
            group((r.day, r.value) for r in ref if r.type == "dividend"),
        ),
        "splits": _match(
            group((s.day, s.factor) for s in splits),
            group((r.day, r.value) for r in ref if r.type == "split"),
        ),
    }


def split_adjusted_close(bars: Sequence[Bar], splits: Sequence[Split]) -> dict[date, float]:
    """只做拆股调整：D 日收盘价除以执行日晚于 D 的全部拆股比例之积（拆股只取到截止日）。"""
    out: dict[date, float] = {}
    for b in bars:
        if b.close is None:
            continue
        f = 1.0
        for s in splits:
            if s.day > b.day:
                f *= s.factor
        out[b.day] = b.close / f
    return out


def compare_adjusted(mine: Mapping[date, float], adjusted: Sequence[Bar]) -> dict[str, Any]:
    """自算拆股调整收盘价与 adjusted=true 收盘价：最大相对差与超过 1e-6 的天数。"""
    adj = {b.day: b.close for b in adjusted if b.close is not None}
    common = sorted(set(mine) & set(adj))
    rel = {d: abs(mine[d] / adj[d] - 1.0) for d in common if adj[d]}
    worst = max(rel, key=lambda d: rel[d], default=None)
    return {
        "common_days": len(common),
        "date_mismatch": len(set(mine) ^ set(adj)),
        "rel_diff_max": None if worst is None else float(f"{rel[worst]:.6g}"),
        "rel_diff_max_day": None if worst is None else worst.isoformat(),
        "days_over_1e-6": sum(1 for v in rel.values() if v > ADJ_TOL),
    }
