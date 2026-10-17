"""17 §2.1–2.2 / T09 §4.2：事件数据结构；对外一律使用日期。"""

from collections import Counter
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum

from stock_risk.domain.numeric import InvalidInputError


class EventKind(StrEnum):
    PCT = "PCT"
    ATR = "ATR"


@dataclass(frozen=True, slots=True)
class EventSetSpec:
    kind: EventKind
    threshold: Decimal
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EventKind):
            raise InvalidInputError("事件种类必须为 PCT 或 ATR")
        if not isinstance(self.threshold, Decimal) or not self.threshold.is_finite():
            raise InvalidInputError("事件阈值必须是有限 Decimal")
        if self.threshold <= 0 or (self.kind is EventKind.PCT and self.threshold >= 1):
            raise InvalidInputError("事件阈值越界")


@dataclass(frozen=True, slots=True)
class EventParameters:
    rho_end: Decimal
    w_q: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.rho_end, Decimal)
            or not self.rho_end.is_finite()
            or not 0 < self.rho_end <= 1
        ):
            raise InvalidInputError("rho_end 必须是 (0, 1] 内的 Decimal")
        if isinstance(self.w_q, bool) or not isinstance(self.w_q, int) or self.w_q < 1:
            raise InvalidInputError("w_q 必须是正整数")


@dataclass(frozen=True, slots=True)
class PendingInterval:
    start_exclusive: date
    end_inclusive: date
    valid_days: int


@dataclass(frozen=True, slots=True)
class SevereEvent:
    s: date
    p: date
    c_p: Decimal
    e: date | None
    reason: str | None
    confirm: date | None
    flags: frozenset[str]
    report_trough: date | None
    k_star: date | None
    censor: frozenset[str]
    unevaluable: frozenset[str]
    dependency_pending: PendingInterval | None
    label_determinable: date | None
    cap_continuation: bool
    merges: int
    is_main: bool
    is_prefix_complete: bool
    a_s: Decimal | None


@dataclass(frozen=True, slots=True)
class EventChain:
    spec: EventSetSpec
    events: tuple[SevereEvent, ...]

    @property
    def dependency_pending(self) -> tuple[PendingInterval, ...]:
        return tuple(e.dependency_pending for e in self.events if e.dependency_pending is not None)

    @property
    def counts(self) -> dict[str, int]:
        """各原因分别计数，可重叠；不把不可评估事件混入主口径。"""
        result = Counter({"total": len(self.events), "main": 0, "prefix_complete": 0})
        for event in self.events:
            result["main"] += event.is_main
            result["prefix_complete"] += event.is_prefix_complete
            result.update(f"censor:{r}" for r in event.censor)
            result.update(f"unevaluable:{r}" for r in event.unevaluable)
        result["dependency_pending_days"] = sum(p.valid_days for p in self.dependency_pending)
        return dict(result)
