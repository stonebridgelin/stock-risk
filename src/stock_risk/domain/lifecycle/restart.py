"""距上一次解除的有效交易日数（T08a C-06；10 号第 1A.7 节）。纯函数。

口径：
- 只看紧邻的前一次预警，它必须以 RELEASE 结束；否则（断点、截尾或没有前一次）为空；
- 只计有日线的交易日，区间为（上一次解除日, 本次启动日]。

回放输出（episodes.csv 的 days_since_prev_release）、回放汇总（restart_gap_*）与对照回放的
投递负担（短期重启）统一调用本模块。
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from stock_risk.domain.lifecycle.model import DailyRecord, DataStatus, EndReason

_BAR_STATUSES = (DataStatus.HAS_BAR, DataStatus.INSUFFICIENT)


@dataclass(frozen=True, slots=True)
class EpisodeSpan:
    """一次预警的起止（只含本计算需要的字段）。"""

    start_day: date
    end_day: date | None
    end_reason: EndReason | None


def bar_days(records: Sequence[DailyRecord]) -> list[date]:
    """有日线的交易日（有日线可计算或历史段不足），按记录顺序（升序）。"""
    return [r.day for r in records if r.data_status in _BAR_STATUSES]


def days_since_prev_release(
    prev: EpisodeSpan | None, start_day: date, bars: Sequence[date]
) -> int | None:
    """bars：有日线的交易日，升序（见 bar_days）。"""
    if prev is None or prev.end_reason is not EndReason.RELEASE or prev.end_day is None:
        return None
    if prev.end_day >= start_day:
        raise ValueError(f"上一次解除日 {prev.end_day} 不早于本次启动日 {start_day}")
    return bisect_right(bars, start_day) - bisect_right(bars, prev.end_day)


def restart_gaps(spans: Sequence[EpisodeSpan], bars: Sequence[date]) -> list[int | None]:
    """按启动日排序后，逐个给出距紧邻前一次预警解除的有效交易日数（第一个为空）。"""
    ordered = sorted(spans, key=lambda s: s.start_day)
    return [
        days_since_prev_release(ordered[i - 1] if i else None, s.start_day, bars)
        for i, s in enumerate(ordered)
    ]
