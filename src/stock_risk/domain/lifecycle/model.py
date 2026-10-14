"""生命周期的状态、事件与诊断记录（10 号规格第 1.1、1.5 节；12 号第 1.7 节；I19）。

状态对象全部不可变，引擎按值复制（dataclasses.replace）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import IntEnum, StrEnum
from typing import Any


class Level(IntEnum):
    WATCH = 1  # 关注
    ALERT = 2  # 警戒


class Display(StrEnum):
    NORMAL = "正常"
    WATCH = "关注"
    ALERT = "警戒"
    DATA_BREAK = "数据中断"
    INSUFFICIENT = "不可计算"


class DataStatus(StrEnum):
    HAS_BAR = "HAS_BAR"  # 有日线
    MISSING = "MISSING"  # 缺价（m ≤ G）
    DATA_BREAK = "DATA_BREAK"  # 缺价且 m ≥ G + 1
    INSUFFICIENT = "INSUFFICIENT"  # 有日线，历史段不足 W + N 条


class EventType(StrEnum):
    START = "START"
    ESCALATE = "ESCALATE"
    DOWNGRADE = "DOWNGRADE"
    RELEASE = "RELEASE"
    DATA_BREAK = "DATA_BREAK"
    DATA_ISSUE = "DATA_ISSUE"
    STABILIZED = "STABILIZED"
    RECOVERY_OBS = "RECOVERY_OBS"
    RECOVERY_REVOKED = "RECOVERY_REVOKED"
    RECOVERY_FAIL = "RECOVERY_FAIL"


LEVEL_EVENTS = frozenset(
    {EventType.START, EventType.ESCALATE, EventType.DOWNGRADE, EventType.RELEASE}
)
RECOVERY_EVENTS = frozenset(
    {
        EventType.STABILIZED,
        EventType.RECOVERY_OBS,
        EventType.RECOVERY_REVOKED,
        EventType.RECOVERY_FAIL,
    }
)


class ResetReason(StrEnum):
    NEW_LOW = "NEW_LOW"
    ESCALATE = "ESCALATE"
    DATA_GAP = "DATA_GAP"


class EndReason(StrEnum):
    """预警的终止原因。截尾（LISTING_END、EXPLORATION_END）不是事件。"""

    RELEASE = "RELEASE"
    DATA_BREAK = "DATA_BREAK"
    LISTING_END = "LISTING_END"
    EXPLORATION_END = "EXPLORATION_END"


# 证据代码（第 1.3 节）
EVID_A, EVID_B, EVID_C = "a", "b", "c"  # E_down
EVID_A1, EVID_B1 = "a′", "b′"  # E_rel
DIRECT_STRONG_RECOVERY = "DIRECT_STRONG_RECOVERY"


@dataclass(frozen=True, slots=True)
class Event:
    """生命周期事件。identity 为业务身份（不含类型；与 type 一起唯一）；transition 为 T1–T16。"""

    type: EventType
    day: date
    identity: tuple[Any, ...]
    transition: str | None = None
    basis: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.type, *self.identity)


@dataclass(frozen=True, slots=True)
class Episode:
    """进行中的预警（第 1.1 节"预警字段"）。dc、rc、xc 为日末计数。"""

    episode_id: str
    start_day: date
    level: Level
    p_ep: float
    p_ep_day: date
    atr_ep: float
    l_ep: float
    l_ep_day: date
    obs_start: date
    cycle_no: int
    since_obs: int = 0
    run_high: float | None = None  # obs_start 之后各有效日的最高收盘（用于冻结 box_high）
    box_high: float | None = None
    downgraded: bool = False
    h_d: float | None = None
    dc: int = 0
    rc: int = 0
    xc: int = 0
    dc_segment: tuple[frozenset[str], ...] = ()  # 当前 dc 计数段内每日的证据集合
    rc_segment: tuple[frozenset[str], ...] = ()
    obs_n: int = 0  # 本周期已发出的 RECOVERY_OBS 次数；obs_issued = obs_n > 0
    stabilized: bool = False  # 本周期是否已发 STABILIZED
    prev_confirm: bool = False
    entry_path: str = "S"  # v0.8：启动路径 S / R / L，启动时冻结（第 1A.3 节）
    delta_start: float = 0.0  # v0.8：Δ_start = P_ep − C_start，启动时冻结

    @property
    def obs_issued(self) -> bool:
        return self.obs_n > 0


@dataclass(frozen=True, slots=True)
class SecurityState:
    """证券级状态（第 1.1 节）：历史段起点、连续缺价天数、最近一次解除日，以及进行中的预警。"""

    symbol: str
    seg_start: date | None = None
    m: int = 0
    last_release: date | None = None
    episode: Episode | None = None
    display: Display = Display.INSUFFICIENT


@dataclass(frozen=True, slots=True)
class DailyRecord:
    """每日诊断记录，覆盖 03 计划 T04b daily.csv 的固定字段，另含状态字段供不变量与夹具断言。"""

    day: date
    in_replay: bool
    data_status: DataStatus
    m: int
    display: Display
    level: Level | None
    seg_start: date | None
    episode_id: str | None
    p_ep: float | None
    p_ep_day: date | None
    atr_ep: float | None
    l_ep: float | None
    l_ep_day: date | None
    d_t: float | None  # 无预警时
    a_start: float | None
    d_ep: float | None  # 有预警时
    r2_status: str | None
    r2_score: float | None
    r2_triggered: bool | None
    cycle_no: int | None
    obs_start: date | None
    since_obs: int | None
    confirm_method: str
    confirm_line: float | None  # SMA10 / box_high / 前 K 日最高
    confirmed: bool | None
    stable: bool | None
    long_stable: bool | None
    new_low: bool | None
    u: float | None
    strong_threshold: float | None  # 强恢复价格门槛 L_ep + min(u_rel × ATR_ep, φ × (P_ep − L_ep))
    strong: bool | None
    evidence: frozenset[str]  # 警戒时 E_down，关注时 E_rel
    dc_cand: int | None
    rc_cand: int | None
    xc_cand: int | None
    dc: int | None  # 日末
    rc: int | None
    xc: int | None
    h_d: float | None
    giveback: float | None
    box_high: float | None
    downgraded: bool | None
    obs_issued: bool | None
    prev_confirm: bool | None
    last_release: date | None
    reset: bool
    reset_reasons: tuple[ResetReason, ...]
    transition: str | None  # 当日等级或数据转移 T1–T16
    end_reason: EndReason | None  # 当日终止的预警及原因
    # ---- v0.8 诊断字段（第 1A 节；开关关闭时照常计算可得的量） ----
    dd_t: float | None = None  # 1 − C / 短窗口 P^max（无预警时）
    dd_l: float | None = None  # 1 − C / 长窗口 P^max_L（long_path ≠ OFF 且无预警时）
    dd_ep: float | None = None  # 1 − C / P_ep（有预警时）
    s_ok: bool | None = None  # 启动路径是否成立（无预警且可计算时）
    r_ok: bool | None = None
    l_ok: bool | None = None
    r2_filtered: bool | None = None  # R2 触发但被 r2_floor 过滤
    entry_path: str | None = None
    delta_start: float | None = None
    theta: float | None = None  # a′ 回滞的 θ（release_hyst = EFFECTIVE 时）
    atr_rec: float | None = None  # 恢复尺度（rec_scale = PRICE 时）


@dataclass(frozen=True, slots=True)
class StepResult:
    state: SecurityState
    events: tuple[Event, ...]
    record: DailyRecord
