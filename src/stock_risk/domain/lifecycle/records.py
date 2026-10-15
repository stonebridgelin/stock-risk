"""每日诊断记录的组装：以日末状态为底，叠加当日计算值（字段见 model.DailyRecord）。"""

from __future__ import annotations

from datetime import date
from typing import Any

from stock_risk.domain.lifecycle.model import DailyRecord, DataStatus, Episode, SecurityState
from stock_risk.domain.lifecycle.params import EngineParams

# 当日计算值：缺省为空，由调用方按当日情况填写（extra）
_DAY_FIELDS = (
    "d_t",
    "a_start",
    "d_ep",
    "r2_status",
    "r2_score",
    "r2_triggered",
    "confirm_line",
    "confirmed",
    "stable",
    "long_stable",
    "new_low",
    "u",
    "strong_threshold",
    "strong",
    "dc_cand",
    "rc_cand",
    "xc_cand",
    "giveback",
    "transition",
    "end_reason",
    "dd_t",
    "dd_l",
    "dd_ep",
    "s_ok",
    "r_ok",
    "l_ok",
    "r2_filtered",
    "theta",
    "atr_rec",
)
# 取自日末预警的字段（无预警时为空）
_EPISODE_FIELDS = (
    "episode_id",
    "p_ep",
    "p_ep_day",
    "atr_ep",
    "l_ep",
    "l_ep_day",
    "cycle_no",
    "obs_start",
    "since_obs",
    "dc",
    "rc",
    "xc",
    "h_d",
    "box_high",
    "downgraded",
    "obs_issued",
    "prev_confirm",
    "entry_path",
    "delta_start",
)


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
    base: dict[str, Any] = dict.fromkeys(_DAY_FIELDS)
    base.update({f: getattr(ep, f) if ep else None for f in _EPISODE_FIELDS})
    base.update(
        day=day,
        in_replay=in_replay,
        data_status=data_status,
        m=state.m,
        display=state.display,
        level=ep.level if ep else None,
        seg_start=state.seg_start,
        confirm_method=params.candidate.confirm.value,
        evidence=frozenset(),
        last_release=state.last_release,
        reset_reasons=(),
    )
    if extra:
        unknown = set(extra) - set(base)
        if unknown:
            raise KeyError(f"诊断记录没有这些字段: {sorted(unknown)}")
        base.update(extra)
    base["reset"] = bool(base["reset_reasons"])
    return DailyRecord(**base)
