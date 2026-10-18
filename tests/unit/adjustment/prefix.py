"""前缀不变性检查（T11-D0a 第 4.8、9.3 节）。

只用截至 T 的输入（日线 ≤ T、行动的原始除权日 ≤ T）重算，投影为截至 T 的视图，与全量结果的
截至 T 视图逐字节比较（规范 JSON）。截断日只取锚日及以后，包括：每个原始除权日及其前后各一个交易日、
每月最后一个交易日、缺价区间的中间日、历史断点前后、存在待处理行动的日子——这里取锚日至最后日期之间的
全部交易日再加上全部原始除权日，是上述集合的超集。
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from datetime import date

from stock_risk.domain.adjustment import truncate_inputs, view_as_of
from stock_risk.domain.adjustment.inputs import DividendInput, SplitInput, Verification
from stock_risk.domain.corrections import RawPriceCorrection
from stock_risk.domain.raw_text import RawBarText
from tests.unit.adjustment.helpers import CAL, PARAMS, run, view_bytes


def truncation_days(
    bars: Sequence[RawBarText],
    splits: Sequence[SplitInput],
    dividends: Sequence[DividendInput],
    calendar: Sequence[date] = CAL,
) -> list[date]:
    anchor = bars[0].day
    ex = [s.ex_date for s in splits] + [d.ex_date for d in dividends]
    last = max([bars[-1].day, *ex])
    lo = bisect.bisect_left(calendar, anchor)
    hi = bisect.bisect_right(calendar, last)
    days = set(calendar[lo : hi + 1])  # 含最后日期之后的一个交易日
    days |= {d for d in ex if d >= anchor}
    return sorted(days)


def check_prefix(
    bars: Sequence[RawBarText],
    splits: Sequence[SplitInput] = (),
    dividends: Sequence[DividendInput] = (),
    verifications: Sequence[Verification] = (),
    corrections: Sequence[RawPriceCorrection] = (),
    params=PARAMS,  # noqa: ANN001
    calendar: Sequence[date] = CAL,
) -> int:
    """返回检查过的截断日数；任一截断日视图不同即 AssertionError。"""
    full = run(bars, splits, dividends, verifications, corrections, params, calendar)
    days = truncation_days(bars, splits, dividends, calendar)
    for t in days:
        tb, ts, td, tv = truncate_inputs(t, bars, splits, dividends, verifications)
        tc = [c for c in corrections if c.day <= t]
        part = run(tb, ts, td, tv, tc, params, calendar)
        a, b = view_bytes(view_as_of(full, t)), view_bytes(view_as_of(part, t))
        assert a == b, f"截断日 {t} 的视图不同"
    return len(days)
