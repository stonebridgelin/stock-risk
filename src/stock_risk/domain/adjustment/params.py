"""复权计算参数（T11-D0a 第 4.2 节）。全部影响结果的参数都在这个冻结数据类中，并写入数据版本。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

SPEC_VERSION = "D0a-v4"
RULING_FILE = "2026-10-17_发起人裁决_D0a待裁决事项A1至A6.md"
# G 未登记（SR-M1）时预检使用的取值集合；这不是对 G 的批准
UNREGISTERED_GAP_BREAKS: tuple[int, ...] = (10, 20)


class AdjustmentParamError(ValueError):
    """复权参数不合规。"""


@dataclass(frozen=True, slots=True)
class AdjustmentParams:
    atr_windows: tuple[int, ...]  # 由应用层从参数网格、冻结配置与评价参数取值（第 4.7 节）
    gap_breaks: tuple[int, ...] = UNREGISTERED_GAP_BREAKS
    price_lower: Decimal = Decimal("0.01")  # 裁决 A-6，包含端点
    price_upper: Decimal = Decimal("1000000000")  # 裁决 A-6，包含端点
    abnormal_dividend_factor: Decimal = Decimal("0.5")  # 裁决 A-4：b_div < 0.5 判异常
    fallback_max_missing_days: int = 5  # 裁决 A-5（计数基准见第 4.5 节）
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        for name in ("atr_windows", "gap_breaks"):
            values = getattr(self, name)
            if not values or any(isinstance(v, bool) or not isinstance(v, int) for v in values):
                raise AdjustmentParamError(f"{name} 必须是非空的正整数序列: {values!r}")
            if any(v < 1 for v in values) or tuple(sorted(set(values))) != values:
                raise AdjustmentParamError(f"{name} 必须为升序、无重复的正整数: {values!r}")
        if not Decimal(0) < self.price_lower < self.price_upper:
            raise AdjustmentParamError("必须 0 < price_lower < price_upper")
        if self.fallback_max_missing_days < 0:
            raise AdjustmentParamError("fallback_max_missing_days 不能为负")

    def as_record(self) -> dict[str, object]:
        """写入 _dataset.json 的形式（数值为十进制文本）。"""
        return {
            "atr_windows": list(self.atr_windows),
            "gap_breaks": list(self.gap_breaks),
            "price_lower": str(self.price_lower),
            "price_upper": str(self.price_upper),
            "abnormal_dividend_factor": str(self.abnormal_dividend_factor),
            "fallback_max_missing_days": self.fallback_max_missing_days,
            "spec_version": self.spec_version,
        }
