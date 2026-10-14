"""研究样本（股票池）定义。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    """研究样本中的角色（I15）：个股运行生命周期；ETF 只提供背景，不运行生命周期。"""

    SIGNAL_TARGET = "signal_target"
    REFERENCE_ONLY = "reference_only"


@dataclass(frozen=True, slots=True)
class UniverseMember:
    """symbol：保存用的代码（当时的代码）；source_candidates：在数据源依次尝试的代码。"""

    symbol: str
    source_candidates: tuple[str, ...]
    category: str
    role: Role
    note: str = ""

    def __post_init__(self) -> None:
        if not self.symbol or not self.source_candidates:
            raise ValueError("symbol 与 source_candidates 不能为空")
        if not isinstance(self.role, Role):
            raise ValueError(f"{self.symbol}: 角色只允许 {[r.value for r in Role]}")
