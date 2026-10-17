"""T09 §4.2 / D-15：方法的逐日状态与事件；引擎 RunResult 适配。

日末"在警" ⇔ 当日处理后存在进行中的预警（等级为关注或警戒）。以 RELEASE 或 DATA_BREAK 终止预警的
当日不在警；缺价日照引擎冻结值保存（缺价日不在掩码内，由调用方按掩码取舍）。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum

from stock_risk.domain.evaluation.decimal_ratio import decimal_price
from stock_risk.domain.lifecycle.model import EventType
from stock_risk.domain.lifecycle.runner import RunResult
from stock_risk.domain.numeric import InvalidInputError


class AlertLevel(StrEnum):
    WATCH = "WATCH"
    ALERT = "ALERT"


# 通知负担报告的事件类型（17 §3E）；START、ESCALATE 也用于 F
NOTICE_KINDS = (
    "START",
    "ESCALATE",
    "DOWNGRADE",
    "RELEASE",
    "RECOVERY_OBS",
    "RECOVERY_REVOKED",
    "RECOVERY_FAIL",
    "DATA_BREAK",
)


@dataclass(frozen=True, slots=True)
class MethodEvent:
    """方法的一个事件。l_ep：RELEASE、RECOVERY_OBS 时该预警当日的 L_ep（价格，六位小数）。"""

    kind: str
    day: date
    episode_id: str
    l_ep: Decimal | None = None


@dataclass(frozen=True, slots=True)
class MethodSeries:
    """与评价日历逐日对齐：levels[i] 为第 i 日日末等级（None 即不在警），
    episodes[i] 为其预警编号。"""

    name: str
    days: tuple[date, ...]
    levels: tuple[AlertLevel | None, ...]
    episodes: tuple[str | None, ...]
    events: tuple[MethodEvent, ...]

    def __post_init__(self) -> None:
        if not (len(self.days) == len(self.levels) == len(self.episodes)):
            raise InvalidInputError(f"{self.name}: 方法序列须与评价日历逐日对齐")
        for level, episode in zip(self.levels, self.episodes, strict=True):
            if (level is None) != (episode is None):
                raise InvalidInputError(f"{self.name}: 在警日必须有 episode_id，不在警日不得有")
        known = set(self.days)
        for event in self.events:
            if event.day not in known or event.kind not in NOTICE_KINDS:
                raise InvalidInputError(f"{self.name}: 事件不合法 {event}")

    def in_alert(self, i: int) -> bool:
        return self.levels[i] is not None

    def is_alert_level(self, i: int) -> bool:
        return self.levels[i] is AlertLevel.ALERT

    def events_of(self, kind: str) -> tuple[MethodEvent, ...]:
        return tuple(e for e in self.events if e.kind == kind)


def from_run_result(name: str, days: Sequence[date], run: RunResult) -> MethodSeries:
    """引擎方法：等级取 DailyRecord.level，episode_id 取 DailyRecord.episode_id（D-15）。

    run 的结果须恰好覆盖 days（评价日历即回放的观察区间）。
    """
    records = [r.record for r in run.results]
    if tuple(r.day for r in records) != tuple(days):
        raise InvalidInputError(f"{name}: 回放结果的日期与评价日历不一致")
    levels: list[AlertLevel | None] = []
    episodes: list[str | None] = []
    for rec in records:
        level = None if rec.level is None else AlertLevel(rec.level.name)
        levels.append(level)
        episodes.append(rec.episode_id if level is not None else None)
    by_day = {rec.day: rec for rec in records}
    events: list[MethodEvent] = []
    for event in run.events:
        if event.type is EventType.DATA_ISSUE or event.type is EventType.STABILIZED:
            continue
        l_ep = None
        if event.type in (EventType.RELEASE, EventType.RECOVERY_OBS):
            raw = event.basis.get("l_ep", by_day[event.day].l_ep)
            l_ep = decimal_price(raw) if raw is not None else None
        events.append(MethodEvent(event.type.value, event.day, str(event.identity[0]), l_ep))
    return MethodSeries(name, tuple(days), tuple(levels), tuple(episodes), tuple(events))


__all__ = ["AlertLevel", "MethodEvent", "MethodSeries", "NOTICE_KINDS", "from_run_result"]
