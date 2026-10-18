"""T11-D0a 测试的公共构造（第 9.1 节"简写夹具的默认输入"）。

- 分红类型 CD，币种 USD（显式给出，因此不记 CURRENCY_ASSUMED_USD），
  公告日为原始除权日前 20 个交易日；
- 拆股公告日为空（记 DECL_DATE_MISSING，获批假设，资格不受影响）；
- 锚日为第一根日线；不带核实登记与更正；
- "第 n 日"为 2015 年 3 月 2 日起的连续交易日，"缺价"指该交易日没有日线；
- 只给收盘的，开、高、低都等于收盘。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

from stock_risk.domain.adjustment import AdjustmentParams, compute_canonical
from stock_risk.domain.adjustment.inputs import (
    DividendInput,
    SplitInput,
    Verification,
    dividend_from_text,
    parse_verification,
    split_from_text,
)
from stock_risk.domain.adjustment.model import AdjustmentResult
from stock_risk.domain.corrections import RawPriceCorrection
from stock_risk.domain.lifecycle.semantic_digest import canonical
from stock_risk.domain.raw_text import RawBarText

SEC = "X"
SPLIT_SOURCE = "test:actions/X_splits.csv"
DIV_SOURCE = "test:actions/X_dividends.csv"
SPLIT_SHA = "a" * 64
DIV_SHA = "b" * 64
PARAMS = AdjustmentParams(atr_windows=(14, 20))


def weekdays(start: date, end: date) -> tuple[date, ...]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return tuple(out)


# 合成交易日历：工作日（夹具所在的 2015 年 3 月期间没有节假日）
CAL = weekdays(date(2013, 1, 1), date(2016, 12, 30))
_START = CAL.index(date(2015, 3, 2))


def day(n: int) -> date:
    """第 n 日（n 从 1 起）：2015-03-02 起的第 n 个交易日。"""
    return CAL[_START + n - 1]


def trading_before(d: date, n: int) -> date:
    i = next(i for i, x in enumerate(CAL) if x >= d)
    return CAL[i - n]


def bar(
    d: date,
    close: str | float,
    *,
    open_: str | float | None = None,
    high: str | float | None = None,
    low: str | float | None = None,
    volume: str | float = "1000",
) -> RawBarText:
    c = str(close)
    o = c if open_ is None else str(open_)
    h = c if high is None else str(high)
    lo = c if low is None else str(low)
    return RawBarText(d, o, h, lo, c, str(volume), o, h, lo, c)


def split(d: date, frm: str, to: str, row: int = 1) -> SplitInput:
    return split_from_text(d, frm, to, SPLIT_SOURCE, row, SPLIT_SHA)


def div(
    d: date,
    amount: str,
    row: int = 1,
    *,
    dividend_type: str = "CD",
    currency: str | None = "USD",
    declaration: date | None | str = "default",
    masked: str = "",
) -> DividendInput:
    if declaration == "default":
        decl = trading_before(d, 20).isoformat()
    else:
        decl = declaration.isoformat() if isinstance(declaration, date) else ""
    return dividend_from_text(
        d, amount, dividend_type, decl, masked, DIV_SOURCE, row, DIV_SHA, currency=currency
    )


def group_id_div(d: date, currency: str = "USD") -> str:
    return f"{SEC}|DIV|{d.isoformat()}|PER_SHARE|{currency}"


def group_id_split(d: date, row: int = 1) -> str:
    return f"{SEC}|SPLIT|{d.isoformat()}|{row}"


def verification(
    vid: str,
    target_type: str,
    target: str,
    kind: str,
    value: str = "",
    sha: str = "",
) -> Verification:
    return parse_verification(
        {
            "id": vid,
            "target_type": target_type,
            "target": target,
            "source_sha256": sha,
            "kind": kind,
            "value": value,
            "ruling": f"R-{vid}",
            "evidence": f"E-{vid}",
        }
    )


def run(
    bars: Sequence[RawBarText],
    splits: Sequence[SplitInput] = (),
    dividends: Sequence[DividendInput] = (),
    verifications: Sequence[Verification] = (),
    corrections: Sequence[RawPriceCorrection] = (),
    params: AdjustmentParams = PARAMS,
    calendar: Sequence[date] = CAL,
) -> AdjustmentResult:
    return compute_canonical(
        SEC, bars, splits, dividends, calendar, verifications, corrections, params
    )


def f_close(r: AdjustmentResult) -> list[str]:
    return [str(b.f_close) for b in r.bars]


def group(r: AdjustmentResult, gid: str):  # noqa: ANN201
    (g,) = [g for g in r.groups if g.group_id == gid]
    return g


def assumption_codes(r: AdjustmentResult) -> list[str]:
    return sorted(a.code for a in r.assumptions)


def issue_codes(r: AdjustmentResult) -> list[str]:
    return sorted(i.code for i in r.issues)


def view_bytes(view: object) -> bytes:
    return canonical(view).encode("utf-8")
