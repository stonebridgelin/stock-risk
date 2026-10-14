"""研究样本（股票池）定义。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UniverseMember:
    """symbol：保存用的代码（当时的代码）；source_candidates：在数据源依次尝试的代码。"""

    symbol: str
    source_candidates: tuple[str, ...]
    category: str
    note: str = ""

    def __post_init__(self) -> None:
        if not self.symbol or not self.source_candidates:
            raise ValueError("symbol 与 source_candidates 不能为空")
