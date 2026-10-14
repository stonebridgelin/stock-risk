"""数据封存策略（v0.7 §5，封存方式 A，发起人 2026-10-13 批准）。

探索期/开发期：数据截至 2019-12-31（含）。
验证期：2020-01-01 起，封存，探索阶段任何代码不得读取或写入。
此处常量是项目纪律，不是可调参数；修改需要发起人批准并记入开发文档变更记录。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from stock_risk.domain.market_data import PriceHistory

EXPLORATION_END = date(2019, 12, 31)
VALIDATION_START = date(2020, 1, 1)


class SealViolation(RuntimeError):
    """出现了封存验证期的数据。必须停止，不能静默丢弃后继续。"""


def truncate_to_exploration(history: PriceHistory) -> PriceHistory:
    """数据源返回后第一道截断：丢弃验证期的行。"""
    return PriceHistory(
        bars=tuple(b for b in history.bars if b.day <= EXPLORATION_END),
        actions=tuple(a for a in history.actions if a.day <= EXPLORATION_END),
    )


def assert_no_sealed_days(days: Iterable[date], what: str) -> None:
    """统一取数入口第一步（I12）：只看日期，出现验证期日期立即报错，不先过滤。"""
    for d in days:
        if d > EXPLORATION_END:
            raise SealViolation(f"{what} 中出现晚于 {EXPLORATION_END} 的行")


def assert_sealed(history: PriceHistory) -> None:
    """写盘前最后一道检查：仍有验证期数据则直接报错。"""
    last = history.last_day
    if last is not None and last > EXPLORATION_END:
        raise SealViolation(f"出现晚于 {EXPLORATION_END} 的数据: {last}")
    for a in history.actions:
        if a.day > EXPLORATION_END:
            raise SealViolation(f"公司行动晚于 {EXPLORATION_END}: {a.day}")
