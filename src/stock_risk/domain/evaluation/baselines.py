"""17 §6 基线表与细节 / D-28：B1 固定百分比、B2 均线、B3 随机日期。

- B1、B2 与候选使用同一公共可评估掩码：掩码成立前不启动；只在掩码日判断。
- 缺价日冻结全部状态（等级、计数、L_ep 不变）；断点日以 DATA_BREAK 终止预警（不发 RELEASE），
  B1 的 last_release 清空；每日最多一次等级转移；比较按 D-01、D-02 精确进行。
- B3 不是逐日方法，只给每个事件的 S1 期望值，不参与 S2。
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext

from stock_risk.domain.evaluation.decimal_ratio import EVALUATION_CONTEXT, drawdown, q12
from stock_risk.domain.evaluation.method_series import AlertLevel, MethodEvent, MethodSeries
from stock_risk.domain.evaluation.params import PEAK_WINDOW
from stock_risk.domain.evaluation.series import EvalSeries
from stock_risk.domain.numeric import InvalidInputError

B1_WATCH_DD = Decimal("0.10")  # 17 §6：回撤 ≥ 10% 关注
B1_ALERT_DD = Decimal("0.20")  # 17 §6：回撤 ≥ 20% 警戒
SMA_DAYS = 50  # 17 §6：SMA50


@dataclass(frozen=True, slots=True)
class BaselineRun:
    """method 为逐日状态与事件；l_ep[i] 为第 i 日日末的 L_ep（不在警为 None）。"""

    method: MethodSeries
    l_ep: tuple[Decimal | None, ...]


@dataclass(slots=True)
class _Episode:
    episode_id: str
    level: AlertLevel
    l_ep: Decimal
    count: int = 0


# 规则：(第 i 日, 进行中的预警或 None) → (新的预警或 None, 当日事件类型或 None)
_Step = Callable[[int, _Episode | None], tuple[_Episode | None, str | None]]


def _check_d(d: int) -> None:
    if isinstance(d, bool) or not isinstance(d, int) or d < 1:
        raise InvalidInputError("基线的退出天数 d 必须为正整数")


def _drive(series: EvalSeries, name: str, step: _Step, on_break: Callable[[], None]) -> BaselineRun:
    """逐日驱动：断点日终止、缺价日冻结、掩码日交给规则；其余有效日（预热）不得在警。"""
    days = series.calendar.days
    breaks = set(series.break_days)
    levels: list[AlertLevel | None] = []
    episodes: list[str | None] = []
    leps: list[Decimal | None] = []
    events: list[MethodEvent] = []
    current: _Episode | None = None
    for i, day in enumerate(days):
        if day in breaks:
            if current is not None:
                events.append(MethodEvent("DATA_BREAK", day, current.episode_id))
            current = None
            on_break()
        elif series.mask[i]:
            before = current
            current, kind = step(i, current)
            if kind is not None:
                ep = current or before
                assert ep is not None
                l_ep = ep.l_ep if kind == "RELEASE" else None
                events.append(MethodEvent(kind, day, ep.episode_id, l_ep))
        elif series.bars[i] is not None and current is not None:
            raise InvalidInputError(f"{name}: {day} 不在掩码内却在警")
        levels.append(current.level if current else None)
        episodes.append(current.episode_id if current else None)
        leps.append(current.l_ep if current else None)
    method = MethodSeries(name, days, tuple(levels), tuple(episodes), tuple(events))
    return BaselineRun(method, tuple(leps))


def _segment_window(series: EvalSeries, i: int, length: int, floor: date | None) -> list[int]:
    """本历史段内、含第 i 日的最近 length 个有效日（日期 ≥ floor），按时间升序。"""
    segment = series.segment_ids[i]
    found: list[int] = []
    j = i
    while j >= 0 and len(found) < length:
        if series.bars[j] is not None:
            if series.segment_ids[j] != segment:
                break
            if floor is not None and series.calendar.days[j] < floor:
                break
            found.append(j)
        j -= 1
    return found[::-1]


def _close(series: EvalSeries, i: int) -> Decimal:
    value = series.closes[i]
    assert value is not None
    return value


def run_b1(series: EvalSeries, d: int, name: str = "B1") -> BaselineRun:
    """D-28 B1：受限窗口回撤；两段式退出；L_ep = 启动当日所用 P_max 的日期起至今日的最低收盘。"""
    _check_d(d)
    state: dict[str, date | None] = {"last_release": None}
    numbering = [0]

    def p_max(i: int) -> tuple[int, Decimal]:
        window = _segment_window(series, i, PEAK_WINDOW, state["last_release"])
        best = max(window, key=lambda j: (_close(series, j), j))  # 并列取最近
        return best, _close(series, best)

    def step(i: int, ep: _Episode | None) -> tuple[_Episode | None, str | None]:
        close = _close(series, i)
        peak_i, peak = p_max(i)
        dd = drawdown(close, peak)
        if ep is None:
            if dd < B1_WATCH_DD:
                return None, None
            numbering[0] += 1
            level = AlertLevel.ALERT if dd >= B1_ALERT_DD else AlertLevel.WATCH
            l_ep = min(_close(series, j) for j in range(peak_i, i + 1) if series.bars[j])
            return _Episode(f"{name}-{numbering[0]}", level, l_ep), "START"
        ep.l_ep = min(ep.l_ep, close)
        if ep.level is AlertLevel.WATCH:
            if dd >= B1_ALERT_DD:
                ep.level, ep.count = AlertLevel.ALERT, 0
                return ep, "ESCALATE"
            ep.count = ep.count + 1 if dd < B1_WATCH_DD else 0
            if ep.count >= d:
                state["last_release"] = series.calendar.days[i]
                return None, "RELEASE"
            return ep, None
        ep.count = ep.count + 1 if dd < B1_ALERT_DD else 0
        if ep.count >= d:
            ep.level, ep.count = AlertLevel.WATCH, 0
            return ep, "DOWNGRADE"
        return ep, None

    return _drive(series, name, step, lambda: state.update(last_release=None))


def sma50(series: EvalSeries) -> tuple[Decimal | None, ...]:
    """D-28 B2：本历史段最近 50 个有效日收盘之和 ÷ 50，量化到 12 位；不足 50 个为 None。"""
    result: list[Decimal | None] = []
    for i in range(len(series.calendar.days)):
        window = _segment_window(series, i, SMA_DAYS, None) if series.bars[i] else []
        if len(window) < SMA_DAYS:
            result.append(None)
            continue
        with localcontext(EVALUATION_CONTEXT):
            total = sum((_close(series, j) for j in window), Decimal(0))
            result.append(q12(total / Decimal(SMA_DAYS)))
    return tuple(result)


def run_b2(series: EvalSeries, d: int, name: str = "B2") -> BaselineRun:
    """D-28 B2：收盘 < SMA50 启动关注；连续 d 日收盘 > SMA50 解除；只有关注一级。"""
    _check_d(d)
    averages = sma50(series)
    numbering = [0]

    def step(i: int, ep: _Episode | None) -> tuple[_Episode | None, str | None]:
        close, average = _close(series, i), averages[i]
        if average is None:  # 掩码（≥ 83）保证不会出现
            raise InvalidInputError(f"{name}: 掩码日缺少 SMA50")
        if ep is None:
            if close >= average:
                return None, None
            numbering[0] += 1
            return _Episode(f"{name}-{numbering[0]}", AlertLevel.WATCH, close), "START"
        ep.l_ep = min(ep.l_ep, close)
        ep.count = ep.count + 1 if close > average else 0
        if ep.count >= d:
            return None, "RELEASE"
        return ep, None

    return _drive(series, name, step, lambda: None)


def b3_expected_s1(series: EvalSeries, peak: date, trough: date) -> Decimal:
    """D-28 B3：E[S1] = (C_p − C̄) ÷ (C_p − C_k)，C̄ 为 [p, k) 内各有效日收盘的平均（不量化）。"""
    lo, hi = series.index(peak), series.index(trough)
    closes = [c for c in series.closes[lo:hi] if c is not None]
    c_p, c_k = series.close(peak), series.close(trough)
    if not closes or c_p <= c_k:
        raise InvalidInputError("B3 需要 p < k 且 C_p > C_k")
    with localcontext(EVALUATION_CONTEXT):
        mean = sum(closes, Decimal(0)) / Decimal(len(closes))
        return q12((c_p - mean) / (c_p - c_k))


__all__ = [
    "B1_ALERT_DD",
    "B1_WATCH_DD",
    "BaselineRun",
    "b3_expected_s1",
    "run_b1",
    "run_b2",
    "sma50",
]
