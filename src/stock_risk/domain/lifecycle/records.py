"""每日诊断记录的组装：以日末状态为底，叠加当日计算值（字段见 model.DailyRecord）。"""

from __future__ import annotations

from datetime import date
from typing import Any

from stock_risk.domain.lifecycle.model import DailyRecord, DataStatus, Episode, SecurityState
from stock_risk.domain.lifecycle.params import EngineParams


def build_record(
    state: SecurityState,
    day: date,
    params: EngineParams,
    *,
    in_replay: bool,
    data_status: DataStatus,
    ep: Episode | None,
    extra: dict[str, Any] | None = None,
) -> DailyRecord:
    """以日末状态为底，叠加当日计算值。"""
    base: dict[str, Any] = {
        "day": day,
        "in_replay": in_replay,
        "data_status": data_status,
        "m": state.m,
        "display": state.display,
        "level": ep.level if ep else None,
        "seg_start": state.seg_start,
        "episode_id": ep.episode_id if ep else None,
        "p_ep": ep.p_ep if ep else None,
        "p_ep_day": ep.p_ep_day if ep else None,
        "atr_ep": ep.atr_ep if ep else None,
        "l_ep": ep.l_ep if ep else None,
        "l_ep_day": ep.l_ep_day if ep else None,
        "d_t": None,
        "a_start": None,
        "d_ep": None,
        "r2_status": None,
        "r2_score": None,
        "r2_triggered": None,
        "cycle_no": ep.cycle_no if ep else None,
        "obs_start": ep.obs_start if ep else None,
        "since_obs": ep.since_obs if ep else None,
        "confirm_method": params.candidate.confirm.value,
        "confirm_line": None,
        "confirmed": None,
        "stable": None,
        "long_stable": None,
        "new_low": None,
        "u": None,
        "strong_threshold": None,
        "strong": None,
        "evidence": frozenset(),
        "dc_cand": None,
        "rc_cand": None,
        "xc_cand": None,
        "dc": ep.dc if ep else None,
        "rc": ep.rc if ep else None,
        "xc": ep.xc if ep else None,
        "h_d": ep.h_d if ep else None,
        "giveback": None,
        "box_high": ep.box_high if ep else None,
        "downgraded": ep.downgraded if ep else None,
        "obs_issued": ep.obs_issued if ep else None,
        "prev_confirm": ep.prev_confirm if ep else None,
        "last_release": state.last_release,
        "reset": False,
        "reset_reasons": (),
        "transition": None,
        "end_reason": None,
    }
    if extra:
        unknown = set(extra) - set(base)
        if unknown:
            raise KeyError(f"诊断记录没有这些字段: {sorted(unknown)}")
        base.update(extra)
    base["reset"] = bool(base["reset_reasons"])
    return DailyRecord(**base)
