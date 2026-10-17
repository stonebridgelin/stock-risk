"""17 §3、§3B / D-16–D-20：延续到谷底的预警段、四类结果、S1、S1w、首次提醒与谷底日。

两种口径：主口径（已确认、无截尾、无不可评估，按报告谷底）与暂定谷底口径（再加前缀完整的
右端截尾事件，按 k*）。不纳入的事件记 EXCLUDED，并给出原因代码（取自事件的截尾、不可评估与
依赖未决），指标一律为空，不写成 0 或 1。
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext
from enum import StrEnum

from stock_risk.domain.evaluation.decimal_ratio import EVALUATION_CONTEXT, drawdown, ratio
from stock_risk.domain.evaluation.event_model import SevereEvent
from stock_risk.domain.evaluation.method_series import MethodSeries
from stock_risk.domain.evaluation.series import EvalSeries

ZERO, ONE = Decimal(0), Decimal(1)


class Caliber(StrEnum):
    MAIN = "MAIN"  # 主口径：报告谷底
    PROVISIONAL = "PROVISIONAL"  # 暂定谷底口径：再加前缀完整的右端截尾事件，按 k*


class OutcomeClass(StrEnum):
    C = "C"
    N = "N"
    U = "U"
    EXCLUDED = "EXCLUDED"  # 该口径不纳入（原因见 reasons）


class USubtype(StrEnum):
    NEVER = "NEVER"
    RELEASED_BEFORE_TROUGH = "RELEASED_BEFORE_TROUGH"


@dataclass(frozen=True, slots=True)
class EventOutcome:
    caliber: Caliber
    outcome_class: OutcomeClass
    reasons: frozenset[str]  # EXCLUDED 时的原因；纳入时为空
    trough: date | None
    s1: Decimal | None
    s1w: Decimal | None
    f0: date | None
    u_subtype: USubtype | None
    pre_peak_in_alert: bool | None
    first_start_day: date | None
    first_start_drawdown: Decimal | None
    not_in_alert_days_after_first: int | None
    trough_day_in_alert: bool | None
    trough_day_release: bool | None


def exclusion_reasons(event: SevereEvent) -> frozenset[str]:
    """事件不进主口径的原因：截尾、不可评估（前缀 UNEVALUABLE_）与依赖未决。"""
    reasons = set(event.censor) | {f"UNEVALUABLE_{r}" for r in event.unevaluable}
    if event.dependency_pending is not None:
        reasons.add("DEPENDENCY_PENDING")
    return frozenset(reasons)


def trough_for(event: SevereEvent, caliber: Caliber) -> date | None:
    """该口径下使用的谷底；不纳入时为空。"""
    if event.is_main:
        return event.report_trough
    if caliber is Caliber.PROVISIONAL and event.is_prefix_complete:
        return event.k_star
    return None


def classify(
    series: EvalSeries, method: MethodSeries, event: SevereEvent, caliber: Caliber
) -> EventOutcome:
    trough = trough_for(event, caliber)
    if trough is None:
        reasons = exclusion_reasons(event)
        return EventOutcome(caliber, OutcomeClass.EXCLUDED, reasons, *([None] * 11))
    with localcontext(EVALUATION_CONTEXT):  # 价格差在固定上下文中计算（D-01）
        return _Classifier(series, method, event, trough).run(caliber)


class _Classifier:
    """在有效日序号上计算；对外字段一律为日期。"""

    def __init__(
        self, series: EvalSeries, method: MethodSeries, event: SevereEvent, trough: date
    ) -> None:
        self.s, self.m, self.e = series, method, event
        self.p = _ordinal(series, event.p)
        self.k = _ordinal(series, trough)
        self.trough = trough

    def cal(self, o: int) -> int:
        """有效日序号 → 日历下标。"""
        return self.s.index(self.s.valid_days[o])

    def alert(self, o: int) -> bool:
        return self.m.in_alert(self.cal(o))

    def close(self, o: int) -> Decimal:
        close = self.s.closes[self.cal(o)]
        assert close is not None
        return close

    def run(self, caliber: Caliber) -> EventOutcome:
        f0 = self.segment_start()
        c_p, c_k = self.e.c_p, self.close(self.k)
        if f0 is None:
            cls, s1, sub = OutcomeClass.U, ONE, self.u_subtype()
        elif f0 < self.p:
            cls, s1, sub = OutcomeClass.C, ZERO, None
        else:
            cls, s1, sub = OutcomeClass.N, ratio(c_p - self.close(f0), c_p - c_k), None
        pre_peak = self.p > 0 and self.alert(self.p - 1)
        first = None if pre_peak else self.first_start()
        k_cal = self.cal(self.k)
        return EventOutcome(
            caliber=caliber,
            outcome_class=cls,
            reasons=frozenset(),
            trough=self.trough,
            s1=s1,
            s1w=self.s1w(f0, c_p, c_k),
            f0=self.s.valid_days[f0] if f0 is not None else None,
            u_subtype=sub,
            pre_peak_in_alert=pre_peak,
            first_start_day=self.s.valid_days[first] if first is not None else None,
            first_start_drawdown=drawdown(self.close(first), c_p) if first is not None else None,
            not_in_alert_days_after_first=self.not_in_alert_after(pre_peak, first),
            trough_day_in_alert=self.m.in_alert(k_cal),
            trough_day_release=any(e.day == self.trough for e in self.m.events_of("RELEASE")),
        )

    def segment_start(self) -> int | None:
        """D-16：自 k − 1 向前，连续在警且 episode_id 相同的最长一段的首日。"""
        if self.k == 0 or not self.alert(self.k - 1):
            return None
        episode = self.m.episodes[self.cal(self.k - 1)]
        o = self.k - 1
        while o > 0 and self.alert(o - 1) and self.m.episodes[self.cal(o - 1)] == episode:
            o -= 1
        return o

    def u_subtype(self) -> USubtype:
        """D-19：p 之前最近的有效日至 k − 1 之间有任何一日在警的，记谷底前解除。"""
        lo = max(self.p - 1, 0)
        if any(self.alert(o) for o in range(lo, self.k)):
            return USubtype.RELEASED_BEFORE_TROUGH
        return USubtype.NEVER

    def first_start(self) -> int | None:
        """D-18：[p, k) 内第一个"当日在警、而前一有效日不在警"的日子。"""
        for o in range(self.p, self.k):
            if self.alert(o) and (o == 0 or not self.alert(o - 1)):
                return o
        return None

    def not_in_alert_after(self, pre_peak: bool, first: int | None) -> int | None:
        """D-18：日末不在警的可评估日数。

        峰值前已在警的区间为 [p, k − 1]，否则为 (首次 START, k − 1]。"""
        if pre_peak:
            lo = self.p
        elif first is not None:
            lo = first + 1
        else:
            return None
        return sum(1 for o in range(lo, self.k) if self.s.mask[self.cal(o)] and not self.alert(o))

    def s1w(self, f0: int | None, c_p: Decimal, c_k: Decimal) -> Decimal:
        """D-17：预警段 [f0, k − 1] 内首个警戒日 f_w。

        f_w < p 记 0，否则为比例；从未警戒或没有预警段记 1。"""
        if f0 is None:
            return ONE
        for o in range(f0, self.k):
            if self.m.is_alert_level(self.cal(o)):
                return ZERO if o < self.p else ratio(c_p - self.close(o), c_p - c_k)
        return ONE


def _ordinal(series: EvalSeries, day: date) -> int:
    o = series.valid_ordinals[series.index(day)]
    assert o is not None, f"{day} 不是有效日"
    return o


__all__ = [
    "Caliber",
    "EventOutcome",
    "OutcomeClass",
    "USubtype",
    "classify",
    "exclusion_reasons",
    "trough_for",
]
