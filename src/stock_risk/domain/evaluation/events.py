"""17 §2.1–2.2 / D-08–D-14：价格事件链、回放、截尾与标签可确定日。"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext

from stock_risk.domain.evaluation._event_state import Progress, advance, distance
from stock_risk.domain.evaluation.decimal_ratio import EVALUATION_CONTEXT
from stock_risk.domain.evaluation.event_model import (
    EventChain,
    EventKind,
    EventParameters,
    EventSetSpec,
    PendingInterval,
    SevereEvent,
)
from stock_risk.domain.evaluation.params import PEAK_WINDOW
from stock_risk.domain.evaluation.series import EvalSeries
from stock_risk.domain.numeric import InvalidInputError


@dataclass(frozen=True, slots=True)
class _Segment:
    series: EvalSeries
    days: tuple[date, ...]
    positions: tuple[int, ...]
    closes: tuple[Decimal, ...]
    break_day: date | None


def build_events(series: EvalSeries, spec: EventSetSpec, params: EventParameters) -> EventChain:
    """只依赖公共价格序列；每段重置 e_prev，确认后从 e 的次日回放（D-09、D-10）。"""
    events: list[SevereEvent] = []
    with localcontext(EVALUATION_CONTEXT):
        for i, history in enumerate(series.segments):
            next_start = series.segments[i + 1].start if i + 1 < len(series.segments) else None
            break_day = next(
                (
                    d
                    for d in series.break_days
                    if d > history.end and (next_start is None or d < next_start)
                ),
                None,
            )
            segment = _Segment(
                series,
                history.bar_days,
                tuple(series.index(d) for d in history.bar_days),
                tuple(series.close(d) for d in history.bar_days),
                break_day,
            )
            events.extend(_scan(segment, spec, params))
    return EventChain(spec, tuple(events))


def _start(
    segment: _Segment, j: int, previous: Progress | None, spec: EventSetSpec
) -> Progress | None:
    if not segment.series.mask[segment.positions[j]]:
        return None
    e_prev = previous.e if previous is not None else None
    lo = max(0, j - PEAK_WINDOW + 1, e_prev if e_prev is not None else 0)
    p = max(range(lo, j + 1), key=lambda k: (segment.closes[k], k))
    atr = segment.series.atr20[segment.positions[j]] if spec.kind is EventKind.ATR else None
    if spec.kind is EventKind.ATR and atr is None:
        raise InvalidInputError(f"{segment.days[j]} 的掩码已成立，但 ATR20 不足")
    if distance(segment.closes[j], segment.closes[p], spec, atr) < spec.threshold:
        return None
    low = min(segment.closes[p : j + 1])
    anchor = max(k for k in range(p, j + 1) if segment.closes[k] == low)
    continuation = (
        previous is not None
        and previous.reason == "c"
        and e_prev is not None
        and j - e_prev <= PEAK_WINDOW
    )
    return Progress(j, p, low, anchor, atr, continuation)


def _scan(segment: _Segment, spec: EventSetSpec, params: EventParameters) -> list[SevereEvent]:
    events: list[SevereEvent] = []
    current: Progress | None = None
    previous: Progress | None = None
    latest: date | None = None
    j = 0
    while j < len(segment.days):
        if current is None:
            current = _start(segment, j, previous, spec)
            j += 1
            continue  # 开始日不判断任何结束条件（D-08）
        current = advance(current, j, segment.closes, spec, params)
        if current.confirm is not None:
            confirmed = segment.days[current.confirm]
            latest = max(latest, confirmed) if latest is not None else confirmed
            events.append(_label(segment, current, latest))
            assert current.e is not None
            j = current.e + 1
            previous, current = current, None
        else:
            j += 1
    if current is not None:
        events.append(_label(segment, current, None))
    return events


def _trough(segment: _Segment, p: int, end: int) -> int:
    """D-12：相等最低收盘的最早日，独立于恢复锚。"""
    return min(range(p, end + 1), key=lambda k: segment.closes[k])


def _unevaluable(segment: _Segment, p: int, trough: int | None) -> frozenset[str]:
    if trough is None:
        return frozenset()
    reasons: set[str] = set()
    for i in range(segment.positions[p], segment.positions[trough] + 1):
        if segment.series.bars[i] is None:
            reasons.add("MISSING")
        elif not segment.series.mask[i]:
            reasons.add("WARMUP")
    return frozenset(reasons)


def _censor(segment: _Segment, state: Progress) -> frozenset[str]:
    lo = segment.positions[state.p]
    if state.confirm is None:
        if segment.break_day is not None:
            return frozenset({"DATA_BREAK"})  # 构成断点的连续缺价不另记 MISSING
        right = f"RIGHT_{segment.series.end_reason.value}"
        # T09-E1R R2（D-11 补充）：未确认事件在 [p, 数据终点] 内有缺价日（含尾部缺价）记 MISSING
        if any(b is None for b in segment.series.bars[lo:]):
            return frozenset({right, "MISSING"})
        return frozenset({right})
    hi = segment.positions[state.confirm]
    return (
        frozenset({"MISSING"})
        if any(b is None for b in segment.series.bars[lo : hi + 1])
        else frozenset()
    )


def _label(segment: _Segment, state: Progress, latest: date | None) -> SevereEvent:
    confirmed = state.confirm is not None
    right = not confirmed and segment.break_day is None
    report = _trough(segment, state.p, state.e) if confirmed and state.e is not None else None
    k_star = _trough(segment, state.p, len(segment.days) - 1) if right else None
    censor = _censor(segment, state)
    unevaluable = _unevaluable(segment, state.p, k_star if right else report)
    pending = None
    if right and state.e is not None:
        pending = PendingInterval(
            segment.days[state.e],
            segment.series.calendar.last,
            len(segment.days) - state.e - 1,
        )
    return SevereEvent(
        s=segment.days[state.s],
        p=segment.days[state.p],
        c_p=segment.closes[state.p],
        e=segment.days[state.e] if state.e is not None else None,
        reason=state.reason,
        confirm=segment.days[state.confirm] if confirmed else None,
        flags=state.flags,
        report_trough=segment.days[report] if report is not None else None,
        k_star=segment.days[k_star] if k_star is not None else None,
        censor=censor,
        unevaluable=unevaluable,
        dependency_pending=pending,
        label_determinable=latest,
        cap_continuation=state.cap_continuation,
        merges=state.merges,
        is_main=confirmed and not censor and not unevaluable,
        is_prefix_complete=right and len(censor) == 1 and not unevaluable,
        a_s=state.a_s,
    )
