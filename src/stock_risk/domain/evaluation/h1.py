"""17 §4 / D-21、D-22：F_ref 与硬约束 H1。

F_ref 的启动条件判定沿用 T06 V0 回归时的引擎口径（D-04）：调用 `start_paths.evaluate_start`，
参照规则 R0 为网格序号 385 的候选、探索假设、默认容差与 V0 开关；`last_release = p` 使 S 路径的
回看窗口为 [max(p, j − 62), j]。只依赖价格与本事件的峰值，对所有方法共用。
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum

from stock_risk.domain.evaluation.decimal_ratio import drawdown
from stock_risk.domain.evaluation.event_model import SevereEvent
from stock_risk.domain.evaluation.method_series import MethodSeries
from stock_risk.domain.evaluation.outcomes import exclusion_reasons
from stock_risk.domain.evaluation.series import EvalSeries
from stock_risk.domain.indicator_source import ComputedIndicators
from stock_risk.domain.lifecycle.params import (
    EXPLORATION_ASSUMPTIONS,
    V0_SWITCHES,
    EngineParams,
    candidate_grid,
)
from stock_risk.domain.lifecycle.start_paths import evaluate_start

R0_CANDIDATE_ID = 385  # D-21：N = 20、k₁ = 2、atr_basis = PEAK


class H1Status(StrEnum):
    OK = "OK"
    VIOLATION = "VIOLATION"
    NOT_JUDGED = "NOT_JUDGED"  # 主口径事件，F_ref 不存在
    UNKNOWN = "UNKNOWN"  # 截尾、不可评估、依赖未决，或右端截尾而 F_ref 不在 [p, k*) 内


@dataclass(frozen=True, slots=True)
class FRef:
    """F_ref 的计算结果。day 为空：在 [p, k) 内从未成立。searched 为假：事件不判 H1，未计算。"""

    searched: bool
    day: date | None
    path: str | None  # S（S 路径成立）或 R（只有跳空路径成立）
    reasons: frozenset[str]  # searched 为假或 day 为空时的原因


@dataclass(frozen=True, slots=True)
class H1Result:
    f_ref: date | None
    path: str | None
    status: H1Status
    reasons: frozenset[str]
    drawdown_at_f_ref: Decimal | None  # F_ref 当日的已实现跌幅 1 − C / C_p


def reference_params() -> EngineParams:
    """R0：网格序号 385、EXPLORATION_ASSUMPTIONS、默认容差、V0 开关。"""
    return EngineParams(
        dict(candidate_grid())[R0_CANDIDATE_ID], EXPLORATION_ASSUMPTIONS, switches=V0_SWITCHES
    )


def find_f_ref(series: EvalSeries, event: SevereEvent, r0: EngineParams | None = None) -> FRef:
    """主口径事件在 [p, 报告谷底) 内、前缀完整的右端截尾事件在 [p, k*) 内搜索；其余事件不判。"""
    if event.is_main:
        trough = event.report_trough
    elif event.is_prefix_complete:
        trough = event.k_star
    else:
        return FRef(False, None, None, exclusion_reasons(event))
    assert trough is not None
    params = r0 or reference_params()
    src = _segment_source(series, event.p)
    for i in range(series.index(event.p), series.index(trough)):
        day = series.calendar.days[i]
        if not series.mask[i]:
            continue
        decision = evaluate_start(src, day, event.p, params)  # 掩码内抛错即停止（D-21）
        if decision.level is not None:
            return FRef(True, day, "S" if decision.record["s_ok"] else "R", frozenset())
    return FRef(True, None, None, frozenset({"F_REF_ABSENT"}))


def judge_h1(series: EvalSeries, method: MethodSeries, event: SevereEvent, f_ref: FRef) -> H1Result:
    """D-22：F_ref 当日日末在警为 OK，否则 VIOLATION。"""
    if not f_ref.searched:
        return H1Result(None, None, H1Status.UNKNOWN, f_ref.reasons, None)
    if f_ref.day is None:
        status = H1Status.NOT_JUDGED if event.is_main else H1Status.UNKNOWN
        return H1Result(None, None, status, f_ref.reasons, None)
    i = series.index(f_ref.day)
    status = H1Status.OK if method.in_alert(i) else H1Status.VIOLATION
    close = series.closes[i]
    assert close is not None
    return H1Result(f_ref.day, f_ref.path, status, frozenset(), drawdown(close, event.c_p))


def _segment_source(series: EvalSeries, peak: date) -> ComputedIndicators:
    """p 所在历史段的 ComputedIndicators（D-21）。"""
    segment = next(s for s in series.segments if s.start <= peak <= s.end)
    by_day = {b.day: b for b in series.bars if b is not None}
    return ComputedIndicators(series.calendar, [by_day[d] for d in segment.bar_days])
