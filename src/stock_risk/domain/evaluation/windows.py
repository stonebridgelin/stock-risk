"""17 §3.1、§3A、§3B / D-23–D-25：恢复参考点 R_E、S2 / S2⁺、h / h⁺ 与指标的可确定日。

全部在有效日序号上运算（D-05）。数据终点为最后一个有效日。窗口不因历史断点截断：断点之后
新历史段中不在掩码内的有效日是窗口内的不可评估日（D-24）。
"""

from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal, localcontext
from enum import StrEnum

from stock_risk.domain.evaluation.decimal_ratio import EVALUATION_CONTEXT, ratio
from stock_risk.domain.evaluation.event_model import EventChain, SevereEvent
from stock_risk.domain.evaluation.method_series import MethodSeries
from stock_risk.domain.evaluation.series import EvalSeries


class RecoveryStatus(StrEnum):
    FOUND = "FOUND"
    NOT_RECOVERED = "NOT_RECOVERED"
    INSUFFICIENT_OBSERVATION = "INSUFFICIENT_OBSERVATION"


class WindowClass(StrEnum):
    A = "A"  # 完整、可评估日充足
    B = "B"  # 完整、可评估日不足
    C = "C"  # 右端观察不足（终点由数据终点决定）
    D = "D"  # 被下一事件截短（S2）


@dataclass(frozen=True, slots=True)
class WindowParams:
    """D-31 拟定值由调用方显式传入：ρ、Y_cap、Y_max、n_min。"""

    rho: Decimal
    y_cap: int
    y_max: int
    n_min: int


@dataclass(frozen=True, slots=True)
class Recovery:
    status: RecoveryStatus
    day: date | None
    ordinal: int | None


@dataclass(frozen=True, slots=True)
class WindowStats:
    """S2 或 h 的窗口。比率为空时 reasons 写明原因（不写成 0）。"""

    empty: bool
    start: date | None
    end: date | None  # 观察窗口的最后一个有效日
    length: int
    evaluable: int
    alert_evaluable: int
    window_class: WindowClass | None
    main_included: bool
    value: Decimal | None  # S2 或 h
    plus_end_ordinal: int | None  # 补足终点（有效日序号）
    plus_length: int
    plus_included: bool
    value_plus: Decimal | None  # S2⁺ 或 h⁺
    target_end_ordinal: int | None  # 目标窗口终点（S2 为 Ω* 的终点；h 为 t_h）
    truncated_by_next: bool
    reasons: frozenset[str]

    @property
    def complete(self) -> bool:
        """观察到目标终点（A、B 类）。"""
        return not self.empty and self.window_class in (WindowClass.A, WindowClass.B)


@dataclass(frozen=True, slots=True)
class S2Result:
    window: WindowStats
    s_main_included: bool
    s_plus_included: bool


class _Ctx:
    """一个证券、一个方法、一个事件链上的窗口计算。"""

    def __init__(self, series: EvalSeries, method: MethodSeries | None, chain: EventChain) -> None:
        self.s, self.m, self.chain = series, method, chain
        self.data_end = len(series.valid_days) - 1

    def ord(self, day: date) -> int:
        o = self.s.valid_ordinals[self.s.index(day)]
        assert o is not None, f"{day} 不是有效日"
        return o

    def cal(self, o: int) -> int:
        return self.s.index(self.s.valid_days[o])

    def next_peak(self, event: SevereEvent) -> int | None:
        """D-23：同一证券、同一事件集按开始日排序的下一个事件（无论截尾、入选）。"""
        later = [e for e in self.chain.events if e.s > event.s]
        return self.ord(min(later, key=lambda e: e.s).p) if later else None

    def counts(self, lo: int, hi: int) -> tuple[int, int]:
        """[lo, hi] 内可评估日与在警的可评估日。"""
        assert self.m is not None
        idx = [self.cal(o) for o in range(lo, hi + 1)]
        evaluable = [i for i in idx if self.s.mask[i]]
        return len(evaluable), sum(1 for i in evaluable if self.m.in_alert(i))


def recovery_ref(
    series: EvalSeries, chain: EventChain, event: SevereEvent, params: WindowParams
) -> Recovery:
    """D-23：在 (k, min(k + Y_max, 下一事件的峰值日)] 内找第一个回升占比 ≥ ρ 的有效日。"""
    assert event.report_trough is not None, "R_E 只对已确认事件计算"
    ctx = _Ctx(series, None, chain)  # R_E 只依赖价格
    k = ctx.ord(event.report_trough)
    nxt = ctx.next_peak(event)
    hi = k + params.y_max if nxt is None else min(k + params.y_max, nxt)
    c_k = series.closes[ctx.cal(k)]
    assert c_k is not None
    with localcontext(EVALUATION_CONTEXT):
        for o in range(k + 1, min(hi, ctx.data_end) + 1):
            c = series.closes[ctx.cal(o)]
            assert c is not None
            if ratio(c - c_k, event.c_p - c_k) >= params.rho:
                return Recovery(RecoveryStatus.FOUND, series.valid_days[o], o)
    status = (
        RecoveryStatus.INSUFFICIENT_OBSERVATION
        if hi > ctx.data_end
        else RecoveryStatus.NOT_RECOVERED
    )
    return Recovery(status, None, None)


def s2_window(
    series: EvalSeries,
    method: MethodSeries,
    chain: EventChain,
    event: SevereEvent,
    recovery: Recovery,
    params: WindowParams,
) -> S2Result:
    """D-24：目标窗口 Ω* = [R_E, R_E + Y_cap − 1]，截至下一事件峰值日 − 1 与数据终点中较早者。"""
    assert recovery.ordinal is not None, "S2 只对 R_E 已找到的事件计算"
    ctx = _Ctx(series, method, chain)
    r = recovery.ordinal
    t_star = r + params.y_cap - 1
    nxt = ctx.next_peak(event)
    b = nxt - 1 if nxt is not None else None
    if b is not None and r > b:  # 截空
        w = _empty(t_star, "EMPTY_WINDOW", WindowClass.D)
        return S2Result(w, False, False)
    obs_end = min(t_star, ctx.data_end, *(() if b is None else (b,)))
    if obs_end == t_star:
        cls, plus_end = None, t_star  # A 或 B，按可评估日定
    elif b is None or ctx.data_end <= b:
        cls, plus_end = WindowClass.C, t_star if b is None else min(t_star, b)
    else:
        cls, plus_end = WindowClass.D, obs_end
    w = _stats(ctx, r, obs_end, plus_end, cls, t_star, cls is WindowClass.D, params.n_min)
    s_main = w.window_class is WindowClass.A
    s_plus = w.plus_included and w.window_class is not WindowClass.D and plus_end == t_star
    return S2Result(w, s_main, s_plus)


def h_window(
    series: EvalSeries,
    method: MethodSeries,
    chain: EventChain,
    event: SevereEvent,
    recovery: Recovery,
    params: WindowParams,
) -> WindowStats:
    """D-25：t_h = min(k + Y_max, 下一事件峰值日 − 1)；t_h < k 为截空。

    只对 R_E 不是 FOUND 的事件计算。"""
    assert recovery.status is not RecoveryStatus.FOUND and event.report_trough is not None
    ctx = _Ctx(series, method, chain)
    k = ctx.ord(event.report_trough)
    nxt = ctx.next_peak(event)
    t_h = k + params.y_max if nxt is None else min(k + params.y_max, nxt - 1)
    if t_h < k:
        return _empty(t_h, "EMPTY_WINDOW", None)
    complete = ctx.data_end >= t_h
    cls = None if complete else WindowClass.C
    w = _stats(ctx, k, min(t_h, ctx.data_end), t_h, cls, t_h, False, params.n_min)
    main_ok = w.main_included and complete and recovery.status is RecoveryStatus.NOT_RECOVERED
    reasons = set(w.reasons)
    if w.main_included and not main_ok:
        reasons.add(recovery.status.value)
    value = w.value if main_ok else None
    return replace(w, main_included=main_ok, value=value, reasons=frozenset(reasons))


def _stats(
    ctx: _Ctx,
    lo: int,
    hi: int,
    plus_end: int,
    cls: WindowClass | None,
    target_end: int,
    truncated: bool,
    n_min: int,
) -> WindowStats:
    length = hi - lo + 1
    evaluable, alert = ctx.counts(lo, hi)
    if cls is None:
        cls = WindowClass.A if evaluable >= n_min else WindowClass.B
    plus_length = plus_end - lo + 1
    unobserved = plus_length - length
    reasons: set[str] = set()
    main = evaluable >= n_min
    if not main:
        reasons.add("BELOW_N_MIN")  # 可评估日少于 n_min，不计算，单列（17 §3.1、§3A）
    plus_ok = plus_length >= n_min
    with localcontext(EVALUATION_CONTEXT):
        value = ratio(Decimal(alert), Decimal(evaluable)) if main else None
        plus = (
            ratio(Decimal(alert + (length - evaluable) + unobserved), Decimal(plus_length))
            if plus_ok
            else None
        )
    if not plus_ok:
        reasons.add("PLUS_BELOW_N_MIN")
    return WindowStats(
        False,
        ctx.s.valid_days[lo],
        ctx.s.valid_days[hi],
        length,
        evaluable,
        alert,
        cls,
        main,
        value,
        plus_end,
        plus_length,
        plus_ok,
        plus,
        target_end,
        truncated,
        frozenset(reasons),
    )


def _empty(target_end: int, reason: str, cls: WindowClass | None) -> WindowStats:
    return WindowStats(
        True,
        None,
        None,
        0,
        0,
        0,
        cls,
        False,
        None,
        None,
        0,
        False,
        None,
        target_end,
        True,
        frozenset({reason}),
    )
