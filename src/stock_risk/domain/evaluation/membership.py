"""17 §1.2、§3D、§3E、§5.3 / D-14、D-27：入选期间、占用率、F 与通知负担、指标的可确定日与两段归属。

- 入选期间为若干 [起日, 止日) 区间；None 表示不按入选期间筛选（探索集，17 §1.2）。
- 占用率 = 入选期间内、掩码内日子中日末在警的比例；
  F 与通知负担 = 同一批日子上的事件数 × 252 ÷ 日数。
- 日数为 0 时比率为空并给出原因（不写成 0），汇总时单独计数（I-13）。
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext
from enum import StrEnum

from stock_risk.domain.evaluation.decimal_ratio import EVALUATION_CONTEXT, ratio
from stock_risk.domain.evaluation.event_model import EventChain, SevereEvent
from stock_risk.domain.evaluation.method_series import NOTICE_KINDS, MethodSeries
from stock_risk.domain.evaluation.series import EvalSeries
from stock_risk.domain.evaluation.windows import WindowStats
from stock_risk.domain.numeric import InvalidInputError

YEAR_DAYS = 252  # "每证券年"按 252 个可评估日折算（D-27）
F_KINDS = ("START", "ESCALATE")  # 17 §3D：向上事件频次


@dataclass(frozen=True, slots=True)
class Membership:
    periods: tuple[tuple[date, date], ...]  # [起日, 止日)

    def __post_init__(self) -> None:
        if any(lo >= hi for lo, hi in self.periods):
            raise InvalidInputError("入选期间须为非空的 [起日, 止日)")

    def contains(self, day: date) -> bool:
        return any(lo <= day < hi for lo, hi in self.periods)


def _member(membership: Membership | None, day: date) -> bool:
    return membership is None or membership.contains(day)


def included_events(chain: EventChain, membership: Membership | None) -> tuple[SevereEvent, ...]:
    """事件按峰值日纳入；未纳入的事件仍在事件链中，作为相邻事件的窗口边界。"""
    return tuple(e for e in chain.events if _member(membership, e.p))


@dataclass(frozen=True, slots=True)
class Rate:
    value: Decimal | None
    numerator: int
    denominator: int
    reason: str | None  # value 为空的原因


def _rate(numerator: int, denominator: int, scale: int = 1) -> Rate:
    if denominator == 0:
        return Rate(None, numerator, 0, "NO_EVALUABLE_DAYS")
    with localcontext(EVALUATION_CONTEXT):
        return Rate(
            ratio(Decimal(numerator * scale), Decimal(denominator)), numerator, denominator, None
        )


def counted_days(series: EvalSeries, membership: Membership | None) -> list[int]:
    """入选期间内、公共可评估掩码内的日子（日历下标）。"""
    return [
        i
        for i, day in enumerate(series.calendar.days)
        if series.mask[i] and _member(membership, day)
    ]


def occupancy(series: EvalSeries, method: MethodSeries, membership: Membership | None) -> Rate:
    days = counted_days(series, membership)
    return _rate(sum(1 for i in days if method.in_alert(i)), len(days))


@dataclass(frozen=True, slots=True)
class NoticeStats:
    days: int
    counts: dict[str, int]
    per_year: dict[str, Rate]  # 各通知类型每证券年
    f: Rate  # START 与 ESCALATE 合计，每证券年


def notice_stats(
    series: EvalSeries, method: MethodSeries, membership: Membership | None
) -> NoticeStats:
    days = counted_days(series, membership)
    day_set = {series.calendar.days[i] for i in days}
    counts = {
        kind: sum(1 for e in method.events_of(kind) if e.day in day_set) for kind in NOTICE_KINDS
    }
    per_year = {kind: _rate(n, len(days), YEAR_DAYS) for kind, n in counts.items()}
    f = _rate(sum(counts[k] for k in F_KINDS), len(days), YEAR_DAYS)
    return NoticeStats(len(days), counts, per_year, f)


@dataclass(frozen=True, slots=True)
class RateSummary:
    """跨证券汇总前的分母核对（I-13）：比率为空的证券不进中位数，个数与原因单独报告。"""

    defined: tuple[Decimal, ...]
    undefined: int
    reasons: dict[str, int]


def summarize_rates(rates: Iterable[Rate]) -> RateSummary:
    items = list(rates)
    defined = tuple(r.value for r in items if r.value is not None)
    reasons = Counter(r.reason or "" for r in items if r.value is None)
    return RateSummary(defined, len(items) - len(defined), dict(reasons))


class Segment(StrEnum):
    FRONT = "FRONT"
    BACK = "BACK"
    NEITHER = "NEITHER"  # 峰值在前段、所需数据延伸到后段（17 §5.3）


def _ord(series: EvalSeries, day: date) -> int:
    o = series.valid_ordinals[series.index(day)]
    assert o is not None
    return o


def window_determinable(
    series: EvalSeries, chain: EventChain, event: SevereEvent, window: WindowStats
) -> int | None:
    """17 §2.2：指标的可确定日（有效日序号，可越过数据终点）。窗口未被下一事件截短的，取
    max(本事件标签的可确定日, 目标窗口终点)；被截短的，取 max(本事件, 下一事件标签的可确定日)。
    所需标签不可确定时为空。"""
    if event.label_determinable is None:
        return None
    own = _ord(series, event.label_determinable)
    if not window.truncated_by_next:
        assert window.target_end_ordinal is not None
        return max(own, window.target_end_ordinal)
    later = [e for e in chain.events if e.s > event.s]
    nxt = min(later, key=lambda e: e.s) if later else None
    if nxt is None or nxt.label_determinable is None:
        return None
    return max(own, _ord(series, nxt.label_determinable))


def attribute(determinable: int | None, peak: int, front_end: int) -> Segment:
    """17 §5.3：可确定日不晚于前段终点的归前段；峰值在前段而所需数据延伸到后段（或不可确定）的
    两段都不用；峰值在后段的归后段。参数均为有效日序号。"""
    if determinable is not None and determinable <= front_end:
        return Segment.FRONT
    return Segment.NEITHER if peak <= front_end else Segment.BACK


__all__: Sequence[str] = (
    "Membership",
    "NoticeStats",
    "Rate",
    "RateSummary",
    "Segment",
    "attribute",
    "counted_days",
    "included_events",
    "notice_stats",
    "occupancy",
    "summarize_rates",
    "window_determinable",
)
