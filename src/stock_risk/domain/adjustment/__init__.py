"""规范前向复权（T11-D0a；D0a 复权规格 v4；发起人裁决 A-1 至 A-6）。纯函数，不做 I/O。"""

from stock_risk.domain.adjustment.compute import compute_canonical
from stock_risk.domain.adjustment.params import AdjustmentParams
from stock_risk.domain.adjustment.view import truncate_inputs, view_as_of

__all__ = ["AdjustmentParams", "compute_canonical", "truncate_inputs", "view_as_of"]
