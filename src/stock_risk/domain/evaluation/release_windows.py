"""17 §3C / D-26：解除（RELEASE）与恢复观察（RECOVERY_OBS）之后的 Q、Q_L 窗口。

窗口为信号日 r 之后（不含 r）的 Q 个有效日。完整 ⇔ 这些日子都在数据内，且与 r 在同一历史段
（不跨越 DATA_BREAK）。三条不利条件：① 1 − C / C_r ≥ y_pct；
② (C_r − C) / A_r ≥ y_atr（A_r = ATR₂₀(r)，D-03）；③ C < L_ep（价格精确比较）。
① ② 按 D-01 量化到 12 位后比较。y_pct、y_atr 未给定时只报告窗口内的量，不判不利（adverse 为空）。

三个条件分别判定（T09-C1 C-05），每日每个条件为成立、不成立或未知（A_r 不可得时 ② 未知，
L_ep 不可得时 ③ 未知）：
- 任一日任一条件已知成立 → 不利；否则有任何未知 → 未知；否则 → 不利为否；
- 首个不利日 = 已知成立的最早一日 d*；d* 之前某日有未知条件的，首个不利日为空，
  加 FIRST_DAY_UNCERTAIN；
- known_criteria：已观察部分中已知成立过的条件；first_criteria：首个不利日满足的条件；
- 原因字段说明输入状况：ATR_UNAVAILABLE、L_EP_UNAVAILABLE、INCOMPLETE_WINDOW
  无论结论是否确定都记入。
窗口不完整、已观察部分没有已知成立也没有未知的，不利照旧记为否（夹具 B41e、B41f）；
T09-C1 要求改为未知，与既有夹具冲突，见 T09-C1 回执停止项 C1-01。
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
    adverse: bool | None  # 未给阈值时为空；有条件未知且没有已知成立的也为空
    first_adverse_day: date | None
    known_criteria: tuple[str, ...] | None  # 已观察部分中已知成立过的条件
    first_criteria: tuple[str, ...] | None  # 首个不利日满足的条件；首个不利日为空时为空
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
    if l_ep is None:
        reasons.add("L_EP_UNAVAILABLE")
    with localcontext(EVALUATION_CONTEXT):
        a_r = q12(a_raw) if a_raw is not None else None
        closes = [series.closes[series.index(d)] for d in days]
        drops = [drawdown(c, c_r) for c in closes if c is not None]
        atr_drops = [ratio(c_r - c, a_raw) for c in closes if c is not None] if a_raw else []
        verdict = _judge(days, closes, c_r, a_raw, l_ep, thresholds)
    if thresholds is None:
        reasons.add("NO_THRESHOLDS")
    reasons |= verdict.reasons
    return PostWindow(
        signal_day,
        q,
        len(days) == q,
        len(days),
        a_r,
        max(drops) if drops else None,
        max(atr_drops) if atr_drops else None,
        any(c < l_ep for c in closes if c is not None) if l_ep is not None else None,
        verdict.adverse,
        verdict.first_day,
        verdict.known,
        verdict.first,
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


@dataclass(frozen=True, slots=True)
class _Verdict:
    adverse: bool | None
    first_day: date | None
    known: tuple[str, ...] | None
    first: tuple[str, ...] | None
    reasons: frozenset[str]


def _day_conditions(
    c: Decimal, c_r: Decimal, a_r: Decimal | None, l_ep: Decimal | None, th: AdverseThresholds
) -> tuple[bool | None, ...]:
    """一日的三个条件（按 CRITERIA 顺序）：成立、不成立或未知（None）。"""
    return (
        drawdown(c, c_r) >= th.y_pct,
        ratio(c_r - c, a_r) >= th.y_atr if a_r is not None else None,
        c < l_ep if l_ep is not None else None,
    )


def _judge(
    days: list[date],
    closes: list[Decimal | None],
    c_r: Decimal,
    a_r: Decimal | None,
    l_ep: Decimal | None,
    th: AdverseThresholds | None,
) -> _Verdict:
    """T09-C1 C-05：三个条件分别判定后合成。"""
    if th is None:
        return _Verdict(None, None, None, None, frozenset())
    known: set[str] = set()
    first_day: date | None = None
    first: tuple[str, ...] | None = None
    unknown_before = unknown_any = False
    for day, c in zip(days, closes, strict=True):
        assert c is not None
        cond = _day_conditions(c, c_r, a_r, l_ep, th)
        hits = tuple(name for name, ok in zip(CRITERIA, cond, strict=True) if ok is True)
        known.update(hits)
        if hits and first is None and not unknown_before:
            first_day, first = day, hits
        if None in cond:
            unknown_any = True
            if first is None:
                unknown_before = True
    reasons: set[str] = set()
    known_t = tuple(n for n in CRITERIA if n in known)
    if known_t and first_day is None:
        reasons.add("FIRST_DAY_UNCERTAIN")  # 首个已知成立日之前，有一日存在未知条件
    if known_t:
        return _Verdict(True, first_day, known_t, first, frozenset(reasons))
    if unknown_any:
        return _Verdict(None, None, None, None, frozenset(reasons))
    return _Verdict(False, None, None, None, frozenset(reasons))
