"""17 §3C / D-26：解除（RELEASE）与恢复观察（RECOVERY_OBS）之后的 Q、Q_L 窗口。

窗口为信号日 r 之后（不含 r）的 Q 个有效日。完整 ⇔ 这些日子都在数据内，且与 r 在同一历史段
（不跨越 DATA_BREAK）。三条不利条件：① 1 − C / C_r ≥ y_pct；
② (C_r − C) / A_r ≥ y_atr（A_r = ATR₂₀(r)，D-03）；③ C < L_ep（价格精确比较）。
① ② 按 D-01 量化到 12 位后比较。y_pct、y_atr 未给定时只报告窗口内的量，不判不利（adverse 为空）。
"""

from collections.abc import Collection
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext

from stock_risk.domain.evaluation.decimal_ratio import EVALUATION_CONTEXT, drawdown, q12, ratio
from stock_risk.domain.evaluation.series import EvalSeries

CRITERIA = ("PCT", "ATR", "L_EP")


@dataclass(frozen=True, slots=True)
class AdverseThresholds:
    """夹具或登记值；探索期未登记时不传（D-25：只报告，不判不利）。"""

    y_pct: Decimal
    y_atr: Decimal


@dataclass(frozen=True, slots=True)
class PostWindow:
    signal_day: date
    q: int
    complete: bool
    observed: int  # 窗口内已观察的有效日数（同一历史段、数据之内）
    a_r: Decimal | None  # ATR₂₀(r)，量化到 12 位；r 当日不可得时为空
    max_drop_pct: Decimal | None  # 窗口内最大的 1 − C / C_r
    max_drop_atr: Decimal | None  # 窗口内最大的 (C_r − C) / A_r
    below_l_ep: bool | None  # 窗口内是否跌破 L_ep；L_ep 未给出时为空
    adverse: bool | None  # 未给阈值或 A_r、L_ep 不可得时为空
    first_adverse_day: date | None
    criteria: tuple[str, ...] | None  # 首个不利日满足的条件
    fail_in_window: bool | None  # 恢复观察：窗口内是否发生 RECOVERY_FAIL
    reasons: frozenset[str]


def post_signal_window(
    series: EvalSeries,
    signal_day: date,
    q: int,
    l_ep: Decimal | None,
    thresholds: AdverseThresholds | None = None,
    fail_days: Collection[date] | None = None,
) -> PostWindow:
    i_r = series.index(signal_day)
    c_r, segment, a_raw = series.closes[i_r], series.segment_ids[i_r], series.atr20[i_r]
    assert c_r is not None, f"{signal_day} 缺价，不能作为信号日"
    days = _window_days(series, i_r, segment, q)
    reasons: set[str] = set()
    if len(days) < q:
        reasons.add("INCOMPLETE_WINDOW")
    if a_raw is None:
        reasons.add("ATR_UNAVAILABLE")
    with localcontext(EVALUATION_CONTEXT):
        a_r = q12(a_raw) if a_raw is not None else None
        closes = [series.closes[series.index(d)] for d in days]
        drops = [drawdown(c, c_r) for c in closes if c is not None]
        atr_drops = [ratio(c_r - c, a_raw) for c in closes if c is not None] if a_raw else []
        first, crit = _first_adverse(days, closes, c_r, a_raw, l_ep, thresholds)
    judged = thresholds is not None and a_raw is not None and l_ep is not None
    if thresholds is None:
        reasons.add("NO_THRESHOLDS")
    return PostWindow(
        signal_day,
        q,
        len(days) == q,
        len(days),
        a_r,
        max(drops) if drops else None,
        max(atr_drops) if atr_drops else None,
        any(c < l_ep for c in closes if c is not None) if l_ep is not None else None,
        (first is not None) if judged else None,
        first if judged else None,
        crit if judged else None,
        any(d in fail_days for d in days) if fail_days is not None else None,
        frozenset(reasons),
    )


def _window_days(series: EvalSeries, i_r: int, segment: int | None, q: int) -> list[date]:
    """r 之后、与 r 同一历史段的前 q 个有效日。"""
    days: list[date] = []
    for i in range(i_r + 1, len(series.calendar.days)):
        if len(days) == q or series.segment_ids[i] != segment:
            break
        if series.bars[i] is not None:
            days.append(series.calendar.days[i])
    return days


def _first_adverse(
    days: list[date],
    closes: list[Decimal | None],
    c_r: Decimal,
    a_r: Decimal | None,
    l_ep: Decimal | None,
    th: AdverseThresholds | None,
) -> tuple[date | None, tuple[str, ...] | None]:
    if th is None or a_r is None or l_ep is None:
        return None, None
    for day, c in zip(days, closes, strict=True):
        assert c is not None
        hit = (
            drawdown(c, c_r) >= th.y_pct,
            ratio(c_r - c, a_r) >= th.y_atr,
            c < l_ep,
        )
        if any(hit):
            return day, tuple(name for name, ok in zip(CRITERIA, hit, strict=True) if ok)
    return None, None
