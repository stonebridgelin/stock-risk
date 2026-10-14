"""按 G 切分历史段（10 号第 1.2 节 A；T2、T4）。引擎与回放共用。

规则：连续缺价达到 G + 1 天即历史断点；断点之后第一个有日线的日子开始新历史段。
第一条日线开始第一个历史段。结果必须与引擎 A 段的判断一致（有测试比较）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from stock_risk.domain.observation import DayPresence


@dataclass(frozen=True, slots=True)
class HistorySegment:
    start: date  # seg_start
    end: date  # 段内最后一条日线
    bar_days: tuple[date, ...]


def split_segments(presence: Sequence[DayPresence], gap_break: int) -> tuple[HistorySegment, ...]:
    segments: list[HistorySegment] = []
    current: list[date] = []
    m = 0
    for p in presence:
        if not p.has_bar:
            m += 1
            continue
        if current and m >= gap_break + 1:
            segments.append(HistorySegment(current[0], current[-1], tuple(current)))
            current = []
        current.append(p.day)
        m = 0
    if current:
        segments.append(HistorySegment(current[0], current[-1], tuple(current)))
    return tuple(segments)
