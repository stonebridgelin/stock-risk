"""代码清单探查（T08d-1 第 1 节）：封存处理、筛选、汇总与覆盖对照。纯函数。

封存处理在读取清单时逐行完成，结果中不出现晚于 2019-12-31 的日期：
- startDate 晚于 2019-12-31 的行丢弃，不计数、不输出；
- endDate 晚于 2019-12-31 或为空的行，endDate 写为 2019-12-31，另记 active_after_2019 = 是，
  不保留真实结束日期；
- endDate 早于 2008-01-02 的行丢弃（只计数）；
- startDate 为空的行无法判断区间，丢弃（只计数）。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from stock_risk.domain.seal import EXPLORATION_END, assert_dates_not_sealed

PROBE_EARLIEST_END = date(2008, 1, 2)  # 结束日早于此日的代码与探索期无关
SELECTED_EXCHANGES = ("NYSE", "NASDAQ")
SELECTED_ASSET_TYPE = "Stock"
SELECTED_CURRENCY = "USD"
REQUIRED_FIELDS = ("ticker", "exchange", "assetType", "priceCurrency", "startDate", "endDate")


class ProbeFormatError(ValueError):
    """代码清单的格式与约定字段不符（停止条件）。"""


@dataclass(frozen=True, slots=True)
class TickerRow:
    """封存处理后的一行。end 不晚于 2019-12-31；active_after_2019 只说明是否延续到其后。"""

    ticker: str
    exchange: str
    asset_type: str
    currency: str
    start: date
    end: date
    active_after_2019: bool

    def __post_init__(self) -> None:
        assert_dates_not_sealed(f"代码清单 {self.ticker}", self.start, self.end)


@dataclass(slots=True)
class SealedList:
    rows: list[TickerRow] = field(default_factory=list)
    dropped_no_start: int = 0
    dropped_ended_before_2008: int = 0


def _date(text: str, ticker: str, name: str) -> date | None:
    text = text.strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError as e:
        raise ProbeFormatError(f"{ticker}: {name} 不是日期") from e


def check_fields(fieldnames: Sequence[str] | None) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in (fieldnames or ())]
    if missing:
        raise ProbeFormatError(f"代码清单缺少字段: {missing}")


def seal_rows(raw: Iterable[Mapping[str, str]]) -> SealedList:
    """逐行做封存处理。晚于截止日的开始日、结束日在这里丢弃或改写，不向外传出。"""
    out = SealedList()
    for r in raw:
        ticker = (r.get("ticker") or "").strip().upper()
        start = _date(r.get("startDate") or "", ticker, "startDate")
        if start is not None and start > EXPLORATION_END:
            continue  # 不计数、不输出
        if start is None:
            out.dropped_no_start += 1
            continue
        end = _date(r.get("endDate") or "", ticker, "endDate")
        active = end is None or end > EXPLORATION_END
        end = EXPLORATION_END if active else end
        assert end is not None
        if end < PROBE_EARLIEST_END:
            out.dropped_ended_before_2008 += 1
            continue
        out.rows.append(
            TickerRow(
                ticker,
                (r.get("exchange") or "").strip(),
                (r.get("assetType") or "").strip(),
                (r.get("priceCurrency") or "").strip(),
                start,
                end,
                active,
            )
        )
    return out


def is_selected(r: TickerRow) -> bool:
    return (
        r.exchange in SELECTED_EXCHANGES
        and r.asset_type == SELECTED_ASSET_TYPE
        and r.currency == SELECTED_CURRENCY
    )


def summarize(sealed: SealedList, examples: int = 5) -> dict[str, Any]:
    """筛选后（NYSE / NASDAQ、Stock、USD）的计数；另列封存处理后全部行的交易所与资产类型取值。"""
    sel = [r for r in sealed.rows if is_selected(r)]
    by_ticker: dict[str, list[TickerRow]] = {}
    for r in sel:
        by_ticker.setdefault(r.ticker, []).append(r)
    reused = {t: rs for t, rs in sorted(by_ticker.items()) if len(rs) > 1}
    stock_usd_other = Counter(
        r.exchange
        for r in sealed.rows
        if r.asset_type == SELECTED_ASSET_TYPE
        and r.currency == SELECTED_CURRENCY
        and r.exchange not in SELECTED_EXCHANGES
    )
    return {
        "total": len(sel),
        "unique_tickers": len(by_ticker),
        "by_exchange": dict(sorted(Counter(r.exchange for r in sel).items())),
        "by_start_year": dict(sorted(Counter(str(r.start.year) for r in sel).items())),
        "ended_by_2019": sum(1 for r in sel if not r.active_after_2019),
        "active_after_2019": sum(1 for r in sel if r.active_after_2019),
        "reused_tickers": len(reused),
        "reused_examples": {
            t: [[r.exchange, r.start.isoformat(), r.end.isoformat()] for r in rs]
            for t, rs in list(reused.items())[:examples]
        },
        "all_rows_after_seal": len(sealed.rows),
        "all_exchange_values": dict(Counter(r.exchange for r in sealed.rows).most_common()),
        "all_asset_type_values": dict(Counter(r.asset_type for r in sealed.rows).most_common()),
        "stock_usd_other_exchanges": dict(stock_usd_other.most_common()),
        "dropped_no_start_date": sealed.dropped_no_start,
        "dropped_ended_before_2008": sealed.dropped_ended_before_2008,
    }


@dataclass(frozen=True, slots=True)
class CoverageRequest:
    ticker: str
    start: date | None = None
    end: date | None = None

    def __post_init__(self) -> None:
        assert_dates_not_sealed(f"覆盖对照 {self.ticker}", self.start, self.end)
        if self.start and self.end and self.start > self.end:
            raise ValueError(f"{self.ticker}: from 晚于 to")


def coverage(sealed: SealedList, requests: Iterable[CoverageRequest]) -> list[dict[str, Any]]:
    """逐个代码：是否在筛选后的清单中、清单区间是否覆盖所需区间；不在筛选范围内时注明。"""
    selected: dict[str, list[TickerRow]] = {}
    other: dict[str, list[TickerRow]] = {}
    for r in sealed.rows:
        (selected if is_selected(r) else other).setdefault(r.ticker, []).append(r)
    out = []
    for q in requests:
        t = q.ticker.upper()
        rows = selected.get(t, [])
        lo, hi = q.start, q.end
        covered: bool | None = None
        if lo is not None or hi is not None:
            covered = any(
                (lo is None or r.start <= lo) and (hi is None or r.end >= hi) for r in rows
            )
        out.append(
            {
                "ticker": t,
                "from": lo.isoformat() if lo else "",
                "to": hi.isoformat() if hi else "",
                "in_list": bool(rows),
                "covered": "" if covered is None else covered,
                "list_ranges": ";".join(f"{r.start}~{r.end}" for r in rows),
                "other_listing": ";".join(
                    f"{r.exchange}/{r.asset_type}/{r.currency}" for r in other.get(t, [])
                ),
            }
        )
    return out
