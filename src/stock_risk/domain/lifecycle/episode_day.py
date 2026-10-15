"""有预警时一日处理的工作对象与结果（从 episode_step 拆出，T08a D-03）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from stock_risk.domain.lifecycle.model import Episode, Event, ResetReason


@dataclass(frozen=True, slots=True)
class EpisodeDay:
    """C 段一日的结果。released 为真时 episode 为当日被解除的预警。fields 写入诊断记录。"""

    episode: Episode
    events: tuple[Event, ...]
    fields: dict[str, Any]
    transition: str | None
    released: bool


@dataclass(slots=True)
class WorkDay:
    """一日的工作状态（仅在 step_episode 内部使用）。"""

    ep: Episode
    day: date
    close: float
    d_ep: float
    dd_ep: float
    new_low: bool
    giveback: float | None
    snapshot: Episode  # 观察重置前（昨日日末）的预警，用于终止事件的身份
    reasons: list[ResetReason] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)
    confirm: bool = False
    escalated: bool = False
    transition: str | None = None
