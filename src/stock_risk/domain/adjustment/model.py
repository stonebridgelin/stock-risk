"""复权结果的数据类型（T11-D0a 第 4.1、4.4 节）。

日志分两张表：原始行动（每条输入记录一条，不带因子）与计算组（因子只记在这里）。
问题与假设清单的每一条都有"决定日"（day / decision_day）：日线检查为该日线的日期，
计算组为 e′，只与原始记录有关的为原始除权日。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum


class Code(StrEnum):
    # 数据假设与处理规则
    DECL_DATE_MISSING = "DECL_DATE_MISSING"  # 裁决 A-1，获批假设
    CURRENCY_ASSUMED_USD = "CURRENCY_ASSUMED_USD"  # 裁决 A-3，获批假设
    DIV_PREV_FALLBACK = "DIV_PREV_FALLBACK"  # 裁决 A-5，获批假设
    ACTION_DATE_SHIFTED = "ACTION_DATE_SHIFTED"  # 规格第 5.5 节，处理规则
    # 未解决的数据问题（该证券不可正式使用）
    DECL_NOT_PRIOR = "DECL_NOT_PRIOR"
    CURRENCY_NOT_USD = "CURRENCY_NOT_USD"
    DIV_TYPE_UNSUPPORTED = "DIV_TYPE_UNSUPPORTED"
    DIV_AMOUNT_INVALID = "DIV_AMOUNT_INVALID"
    SPLIT_RATIO_INVALID = "SPLIT_RATIO_INVALID"
    UNIT_CONVERSION_FAILED = "UNIT_CONVERSION_FAILED"
    SAME_DAY_BASIS_UNVERIFIED = "SAME_DAY_BASIS_UNVERIFIED"
    SAME_DAY_MULTI_SPLIT = "SAME_DAY_MULTI_SPLIT"
    NO_REFERENCE_CLOSE = "NO_REFERENCE_CLOSE"
    FALLBACK_TOO_FAR = "FALLBACK_TOO_FAR"
    ABNORMAL_FACTOR = "ABNORMAL_FACTOR"
    ILLEGAL_FACTOR = "ILLEGAL_FACTOR"
    REPRESENTABILITY = "REPRESENTABILITY"
    ATR_NONPOSITIVE = "ATR_NONPOSITIVE"


# 发起人 2026-10-17 批准的数据假设：报告中标"获批假设，未经历史验证"
APPROVED_ASSUMPTIONS = frozenset(
    {Code.DECL_DATE_MISSING, Code.CURRENCY_ASSUMED_USD, Code.DIV_PREV_FALLBACK}
)
ASSUMPTION_CODES = APPROVED_ASSUMPTIONS | {Code.ACTION_DATE_SHIFTED}
APPROVED_LABEL = "获批假设，未经历史验证"
RULE_LABEL = "处理规则"


class GroupStatus(StrEnum):
    APPLIED = "APPLIED"
    ANCHOR_NOT_APPLIED = "ANCHOR_NOT_APPLIED"  # 锚日行动，不作用
    PENDING = "PENDING"
    REJECTED = "REJECTED"


class ActionKind(StrEnum):
    SPLIT = "SPLIT"
    DIVIDEND = "DIVIDEND"


class Eligibility(StrEnum):
    USABLE = "USABLE"
    NOT_USABLE = "NOT_USABLE"


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """原始行动表的一行。标记只含由记录本身与登记决定的代码，不带 e′。"""

    action_id: str
    security: str
    action_kind: ActionKind
    source: str
    row: int
    source_sha256: str
    original_ex_date: date
    values: tuple[str, ...]  # 原始数值文本：拆股 (from, to)，分红 (cash_amount,)
    dividend_type: str | None
    declaration_date: date | None
    currency: str | None
    group_id: str | None  # 被拒绝而没有组的为空
    codes: tuple[str, ...]
    verification_ids: tuple[str, ...]
    corrected_declaration_date: date | None


@dataclass(frozen=True, slots=True)
class ReferenceFactor:
    group_id: str
    b: Decimal


@dataclass(frozen=True, slots=True)
class GroupRecord:
    """计算组表的一行。待处理（PENDING）的组只有身份、成员与原始除权日。"""

    group_id: str
    security: str
    action_kind: ActionKind
    members: tuple[str, ...]  # 成员 action_id，排序
    original_ex_date: date
    effective_day: date | None  # e′
    status: GroupStatus
    b: Decimal | None = None
    p: date | None = None
    r_p: Decimal | None = None
    k: int | None = None
    dividend_total: Decimal | None = (
        None  # 组内金额之和 D（同日口径为拆股前每股时另记 D′ 于 c_ref 推导中）
    )
    reference_factors: tuple[ReferenceFactor, ...] = ()
    c_ref: Decimal | None = None
    reference_is_theoretical: bool | None = None
    codes: tuple[str, ...] = ()
    verification_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Assumption:
    security: str
    code: str
    level: str  # action / group
    target_id: str  # action_id 或 group_id
    original_ex_date: date
    effective_day: date | None  # 只有 group 层级才填
    source: str
    k: int | None
    decision_day: date


@dataclass(frozen=True, slots=True)
class Issue:
    """未解决的数据问题。day 即决定日。"""

    security: str
    code: str
    day: date
    field: str
    value: str
    note: str
    level: str  # action / group / bar


@dataclass(frozen=True, slots=True)
class AdjustedBar:
    """一根有效日线：原始价（已作用原始价更正、量化到 6 位）、累计因子与 F、F_V。

    F 无法量化到 6 位（数值过大）时为 None，记可表示性问题。"""

    day: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    b_cum: Decimal
    f_open: Decimal | None
    f_high: Decimal | None
    f_low: Decimal | None
    f_close: Decimal | None
    f_volume: Decimal | None


@dataclass(frozen=True, slots=True)
class AdjustmentResult:
    security: str
    anchor: date
    bars: tuple[AdjustedBar, ...]
    actions: tuple[ActionRecord, ...]
    groups: tuple[GroupRecord, ...]
    assumptions: tuple[Assumption, ...]
    issues: tuple[Issue, ...]
    eligibility: Eligibility
    reasons: tuple[str, ...]  # 不可正式使用的原因代码（排序、去重）
    amplitude_collapse_days: tuple[date, ...]  # 只计数，不取消资格
    terminal_pending: tuple[str, ...]  # 终点待处理的组身份，单独计数
    raw_correction_ids: tuple[str, ...]
    verification_ids_used: tuple[str, ...]
    unrelated_verification_ids: tuple[str, ...]  # 目标不属于本证券的登记，不报错，报告中列出


@dataclass(frozen=True, slots=True)
class AdjustmentView:
    """截至 T 的视图（第 4.8 节）。使用资格是整段历史的结论，不在视图中。"""

    security: str
    anchor: date
    as_of: date
    bars: tuple[AdjustedBar, ...]
    actions: tuple[ActionRecord, ...]
    groups: tuple[GroupRecord, ...]
    assumptions: tuple[Assumption, ...]
    issues: tuple[Issue, ...]
