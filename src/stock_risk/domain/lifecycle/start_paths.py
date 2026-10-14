"""无预警时的启动判断（10 号第 1.2 节 B；v0.8 第 1A.3 节的 S、R、L 三条路径）。

- S：D_T ≥ k₁ 且（x₁ > 0 时）DD_T ≥ x₁；
- R：R2 触发，且（r2_floor = ON 时）DD_T ≥ x₁；
- L：long_path ≠ OFF 且 DD^L_T ≥ x_L。
冻结基准取首条成立的路径（S > R > L）；S、R 用短窗口，L 用长窗口。
初始等级按完整警戒条件（D_ep ≥ k₂ 且（x₂ > 0 时）DD_ep ≥ x₂）。
开关全部关闭时（V0）与 v0.7.1 逐项相同：T6 ⇔ D_T ≥ k₂；T7 ⇔ D_T ≥ k₁ 或 R2。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from stock_risk.domain.indicator_source import IndicatorSource, compute_r2
from stock_risk.domain.lifecycle.model import Level
from stock_risk.domain.lifecycle.params import AtrBasis, EngineParams, LongPath
from stock_risk.domain.numeric import InvalidInputError, Kind, ge


def full_alert(d_ep: float, dd_ep: float, params: EngineParams) -> bool:
    """完整警戒条件：D_ep ≥ k₂ 且（x₂ > 0 时）DD_ep ≥ x₂（均用 ε_r，第 1A.2 节）。"""
    tol, x2 = params.tol, params.switches.x2
    return ge(d_ep, params.candidate.k2, Kind.RATIO, tol) and (
        x2 == 0 or ge(dd_ep, x2, Kind.RATIO, tol)
    )


def _a_start(
    src: IndicatorSource, peak_day: date, day: date, params: EngineParams
) -> tuple[float, float | None]:
    """短窗口的 A_start（第 1.3 节）。返回（A_start, ATR_N(T) 或 None）。"""
    n = params.candidate.n
    atr_peak = src.atr(peak_day, n)
    if atr_peak is None:
        raise InvalidInputError(f"{day}: 峰值日 {peak_day} 的 ATR_{n} 不可计算")
    if params.candidate.atr_basis is AtrBasis.PEAK:
        return atr_peak, None
    atr_t = src.atr(day, n)
    if atr_t is None:
        raise InvalidInputError(f"{day}: ATR_{n} 不可计算")
    return min(atr_peak, atr_t), atr_t


@dataclass(frozen=True, slots=True)
class StartDecision:
    """启动判断的结果。level 为 None 表示不启动。"""

    level: Level | None
    transition: str
    reasons: tuple[str, ...]
    entry_path: str | None
    p_ep: float
    p_ep_day: date
    atr_ep: float
    record: dict[str, Any]  # 写入诊断记录的字段
    basis: dict[str, Any]  # START 依据（v0.7.1 原有字段）
    basis_v08: dict[str, Any] = field(default_factory=dict)  # v0.8 新增的 START 数值


def _long_path(
    src: IndicatorSource, day: date, c: float, last_release: date | None, params: EngineParams
) -> tuple[float, date, float]:
    p_l, p_l_day = src.window_max_close(day, params.assumptions.w_long, last_release)
    return p_l, p_l_day, 1 - c / p_l


def _l_a_start(
    src: IndicatorSource, p_day: date, day: date, params: EngineParams
) -> tuple[float, dict[str, Any]]:
    """L 路径的 A_start：PEAK 分量为 ATR_N(P^max_L 日)，不可得时取 [P^max_L 日, T] 内
    最早一个 ATR 可得日的 ATR（替代尺度）；MIN 时取 min(PEAK 分量, ATR_N(T))。"""
    n = params.candidate.n
    comp = src.atr(p_day, n)
    available = comp is not None
    comp_day: date | None = p_day
    if comp is None:
        comp_day = src.first_atr_day(p_day, day, n)
        comp = src.atr(comp_day, n) if comp_day is not None else None
    if comp is None:
        raise InvalidInputError(f"{day}: 长窗口 [{p_day}, {day}] 内没有可得的 ATR_{n}")
    atr_t = src.atr(day, n)
    selected = "PEAK_COMPONENT"
    a_start = comp
    if params.candidate.atr_basis is AtrBasis.MIN:
        if atr_t is None:
            raise InvalidInputError(f"{day}: ATR_{n} 不可计算")
        if atr_t < comp:
            a_start, selected = atr_t, "T"
    return a_start, {
        "p_ep_day": p_day,
        "peak_atr_available": available,
        "peak_component_atr_day": comp_day,
        "peak_component_atr": comp,
        "atr_t": atr_t,
        "a_start_selected": selected,
    }


def evaluate_start(
    src: IndicatorSource, day: date, last_release: date | None, params: EngineParams
) -> StartDecision:
    cand, tol, sw = params.candidate, params.tol, params.switches
    c = src.close(day)
    p_max, p_day = src.window_max_close(day, params.fixed.w, last_release)
    a_start, _ = _a_start(src, p_day, day, params)
    d_t = (p_max - c) / a_start
    dd_t = 1 - c / p_max
    r2 = compute_r2(src, day, cand.n, params.fixed.g, tol)
    floor_ok = sw.x1 == 0 or ge(dd_t, sw.x1, Kind.RATIO, tol)
    s_ok = ge(d_t, cand.k1, Kind.RATIO, tol) and floor_ok
    r_ok = r2.triggered and (not sw.r2_floor or ge(dd_t, sw.x1, Kind.RATIO, tol))
    dd_l: float | None = None
    l_ok = False
    if sw.long_path is not LongPath.OFF:
        p_l, p_l_day, dd_l = _long_path(src, day, c, last_release, params)
        l_ok = ge(dd_l, sw.x_long, Kind.RATIO, tol)
    record: dict[str, Any] = {
        "d_t": d_t,
        "a_start": a_start,
        "r2_status": r2.status.value,
        "r2_score": r2.gap_score,
        "r2_triggered": r2.triggered,
        "dd_t": dd_t,
        "dd_l": dd_l,
        "s_ok": s_ok,
        "r_ok": r_ok,
        "l_ok": l_ok,
        "r2_filtered": r2.triggered and not r_ok,
    }
    basis = {"r2_status": r2.status.value, "r2_score": r2.gap_score}
    if not (s_ok or r_ok or l_ok):
        return StartDecision(None, "T8", (), None, p_max, p_day, a_start, record, basis)

    v08: dict[str, Any] = {"dd_t": dd_t, "dd_l": dd_l, "x1": sw.x1, "x2": sw.x2}
    if s_ok or r_ok:
        path, p_ep, p_ep_day, atr_ep = ("S" if s_ok else "R"), p_max, p_day, a_start
    else:
        path, p_ep, p_ep_day = "L", p_l, p_l_day
        atr_ep, extra = _l_a_start(src, p_l_day, day, params)
        v08.update(extra)
    d_ep, dd_ep = (p_ep - c) / atr_ep, 1 - c / p_ep
    level = Level.ALERT if full_alert(d_ep, dd_ep, params) else Level.WATCH
    reasons: list[str] = []
    if s_ok:
        reasons.append("R1_GE_K2" if level is Level.ALERT else "R1_GE_K1")
    if r_ok:
        reasons.append("GAP")
    if not reasons:
        reasons.append("R1L_CUMULATIVE")
    v08["entry_path"] = path
    record.update(d_ep=d_ep, dd_ep=dd_ep)
    basis.update(d_t=d_t, a_start=atr_ep, p_ep=p_ep, p_ep_day=p_ep_day)
    return StartDecision(
        level,
        "T6" if level is Level.ALERT else "T7",
        tuple(reasons),
        path,
        p_ep,
        p_ep_day,
        atr_ep,
        record,
        basis,
        v08,
    )
