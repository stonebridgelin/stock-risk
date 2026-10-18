"""截至 T 的视图与截断输入（T11-D0a 第 4.8 节；规格第 10 节第 1 项）。

原则：每个字段都按"截至 T 能否确定"投影，不能确定的一律为空。
- 原始行动：原始除权日 > T 的不出现；其余照原样（标记只由记录本身决定，不带 e′）。
- 计算组：原始除权日 > T 的不出现；e′ ≤ T 的照原样；原始除权日 ≤ T 而 e′ > T（或 e′ 不存在）的
  投影为 PENDING，只保留身份、成员与原始除权日。
- 问题与假设：只保留决定日 ≤ T 的条目。
- 使用资格是整段历史的结论，不在视图中。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from stock_risk.domain.adjustment.inputs import (
    DividendInput,
    SplitInput,
    TargetType,
    Verification,
)
from stock_risk.domain.adjustment.model import (
    AdjustmentResult,
    AdjustmentView,
    GroupRecord,
    GroupStatus,
)
from stock_risk.domain.raw_text import RawBarText


class ViewError(ValueError):
    """截断日早于锚日（G-T11-06：截断日只取锚日及以后）。"""


def _project_group(g: GroupRecord, as_of: date) -> GroupRecord:
    if g.effective_day is not None and g.effective_day <= as_of:
        return g
    if g.status is GroupStatus.PENDING and g.effective_day is None:
        return g
    return GroupRecord(
        g.group_id,
        g.security,
        g.action_kind,
        g.members,
        g.original_ex_date,
        None,
        GroupStatus.PENDING,
    )


def view_as_of(result: AdjustmentResult, as_of: date) -> AdjustmentView:
    if as_of < result.anchor:
        raise ViewError(f"{result.security}: 截断日 {as_of} 早于锚日 {result.anchor}")
    return AdjustmentView(
        security=result.security,
        anchor=result.anchor,
        as_of=as_of,
        bars=tuple(b for b in result.bars if b.day <= as_of),
        actions=tuple(a for a in result.actions if a.original_ex_date <= as_of),
        groups=tuple(
            _project_group(g, as_of) for g in result.groups if g.original_ex_date <= as_of
        ),
        assumptions=tuple(a for a in result.assumptions if a.decision_day <= as_of),
        issues=tuple(i for i in result.issues if i.day <= as_of),
    )


def truncate_inputs(
    as_of: date,
    bars_text: Sequence[RawBarText],
    splits: Sequence[SplitInput],
    dividends: Sequence[DividendInput],
    verifications: Sequence[Verification],
) -> tuple[
    tuple[RawBarText, ...],
    tuple[SplitInput, ...],
    tuple[DividendInput, ...],
    tuple[Verification, ...],
]:
    """截至 T 的输入：日线 ≤ T、行动的原始除权日 ≤ T；核实登记只保留目标仍在输入中的
    （目标行动在 T 之后的登记，截至 T 时没有对象）。"""
    bars = tuple(b for b in bars_text if b.day <= as_of)
    sp = tuple(s for s in splits if s.ex_date <= as_of)
    dv = tuple(d for d in dividends if d.ex_date <= as_of)
    action_ids = {x.action_id for x in (*sp, *dv)}
    dropped_ids = {x.action_id for x in (*splits, *dividends)} - action_ids
    kept: list[Verification] = []
    for v in verifications:
        if v.target_type is TargetType.ACTION:
            if v.target in dropped_ids:
                continue
        else:
            parts = v.target.split("|")
            if len(parts) > 2 and date.fromisoformat(parts[2]) > as_of:
                continue
        kept.append(v)
    return bars, sp, dv, tuple(kept)
