"""有预警时用到的规则片段：观察重置、证据排序、恢复确认（10 号第 1.2、1.3 节）。"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from stock_risk.domain.indicator_source import IndicatorSource
from stock_risk.domain.lifecycle.model import (
    EVID_A,
    EVID_A1,
    EVID_B,
    EVID_B1,
    EVID_C,
    Episode,
)
from stock_risk.domain.lifecycle.params import Confirm, EngineParams, ReleaseHyst
from stock_risk.domain.numeric import InvalidInputError, Kind, gt, lt

EVIDENCE_ORDER = (EVID_A, EVID_B, EVID_C, EVID_A1, EVID_B1)
REVOKE_REASON = {
    Confirm.SMA: "BELOW_SMA",
    Confirm.BOX: "BACK_INTO_BOX",
    # 规格未给出 BOX_ROLL 的撤销原因代码；临时代码，待产品经理裁定（T02-T03 回执停止项）
    Confirm.BOX_ROLL: "BELOW_ROLLING_HIGH",
}


def ordered(evidence: frozenset[str] | set[str]) -> list[str]:
    return [e for e in EVIDENCE_ORDER if e in evidence]


def observation_reset(ep: Episode, day: date) -> Episode:
    """观察重置（第 1.2 节）：obs_start = T；box_high、obs_issued 清空；三计数清零；
    cycle_no 在上一交易日的值上加一。同一交易日只执行一次由调用方保证。"""
    return replace(
        ep,
        obs_start=day,
        cycle_no=ep.cycle_no + 1,
        since_obs=0,
        run_high=None,
        box_high=None,
        obs_n=0,
        stabilized=False,
        dc=0,
        rc=0,
        xc=0,
        dc_segment=(),
        rc_segment=(),
        xc_segment=(),
    )


def confirmation(
    ep: Episode, day: date, close: float, src: IndicatorSource, params: EngineParams
) -> tuple[bool, float | None]:
    """恢复确认与确认线（第 1.3 节）。返回（是否成立，确认线数值）。

    SMA：企稳且 C > SMA10 + ε_p。BOX：since_obs > K 且 C > box_high + ε_p。
    BOX_ROLL：since_obs > K 且 C > 今日之前最近 K 个有效交易日的最高收盘 + ε_p。
    """
    k, method, tol = params.candidate.k, params.candidate.confirm, params.tol
    if method is Confirm.SMA:
        line = src.sma(day, params.fixed.sma_len)
        stable = ep.since_obs >= k
        if stable and line is None:
            raise InvalidInputError(f"{day}: 企稳日缺少 SMA{params.fixed.sma_len}")
        return stable and line is not None and gt(close, line, Kind.PRICE, tol), line
    line = ep.box_high if method is Confirm.BOX else src.max_close_prior(day, k)
    if ep.since_obs <= k:
        return False, line
    if line is None:  # T08a B-02：应有确认线时缺线，不得静默视为不成立
        raise InvalidInputError(f"{day}: {method.value} 的确认线缺失（since_obs = {ep.since_obs}）")
    return gt(close, line, Kind.PRICE, tol), line


def theta(ep: Episode, params: EngineParams) -> float | None:
    """a′ 回滞的 θ = min(E_entry, Δ_start)（第 1A.5 节）；release_hyst = OLD 时为 None。

    E_entry：S、R 路径为 E₁ = max(k₁ × ATR_ep, x₁ × P_ep)；L 路径为 x_L × P_ep。
    """
    sw = params.switches
    if sw.release_hyst is ReleaseHyst.OLD:
        return None
    if ep.entry_path == "L":
        e_entry = sw.x_long * ep.p_ep
    else:
        e_entry = max(params.candidate.k1 * ep.atr_ep, sw.x1 * ep.p_ep)
    return min(e_entry, ep.delta_start)


def a_prime(ep: Episode, close: float, d_ep: float, params: EngineParams) -> bool:
    """E_rel 的 a′：OLD 为 D_ep < k₁（ε_r）；EFFECTIVE 为 Δ < h × θ（价格比较，ε_p）。"""
    th = theta(ep, params)
    if th is None:
        return lt(d_ep, params.candidate.k1, Kind.RATIO, params.tol)
    return lt(ep.p_ep - close, params.assumptions.h * th, Kind.PRICE, params.tol)
