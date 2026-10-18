"""复权的行动输入、核实登记与原始价更正（T11-D0a 第 3、5、6 节；规格第 8 节）。

行动的数值从 CSV 原文在 PARSE_CONTEXT 中解析为 Decimal，不经过 float。原文无法解析、非有限的，
数值记为 None，由复权计算记为未解决的数据问题，而不是在这里报错（规格第 4、6 节）。

人工配置（核实登记、原始价更正）先判日期、再解析数值：日期晚于 2019-12-31 → SealViolation。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext
from enum import StrEnum

from stock_risk.domain.numeric import PARSE_CONTEXT, InvalidInputError, _decimal
from stock_risk.domain.seal import assert_dates_not_sealed

PER_SHARE = "PER_SHARE"  # 现有来源没有单位列，固定为每股（G-T11-02）
NONE_CURRENCY = "NONE"  # 组身份中"数据没有币种字段"的写法


class AdjustmentInputError(ValueError):
    """复权输入（核实登记、原始价更正、行动身份）不合规。整次构建停止。"""


def parse_decimal_or_none(text: str | None) -> Decimal | None:
    """原文 → Decimal；空、无法解析、非有限 → None（交由复权计算记为数据问题）。"""
    if text is None or not text.strip():
        return None
    with localcontext(PARSE_CONTEXT):
        try:
            return _decimal(text, "行动数值")
        except InvalidInputError:
            return None


def _action_id(source: str, row: int) -> str:
    return f"{source}#{row}"


@dataclass(frozen=True, slots=True)
class SplitInput:
    """一次拆股（Massive 的 execution_date 即除权日）。b = split_from ÷ split_to。"""

    ex_date: date
    split_from: Decimal | None
    split_to: Decimal | None
    source: str  # 例如 massive:actions/<T>_splits.csv
    row: int  # 数据行号：表头之后从 1 起
    source_sha256: str
    raw_values: tuple[str, str] = ("", "")  # (split_from, split_to) 原文
    declaration_date: date | None = None  # 数据中没有拆股公告日，恒为 None

    @property
    def action_id(self) -> str:
        return _action_id(self.source, self.row)


@dataclass(frozen=True, slots=True)
class DividendInput:
    """一笔现金分红。currency 为 None 表示数据中没有币种字段（裁决 A-3）。"""

    ex_date: date
    cash_amount: Decimal | None
    dividend_type: str
    declaration_date: date | None
    declaration_masked: bool  # masked_after_2019 含 declaration_date：公告日晚于 2019-12-31
    currency: str | None
    share_unit: str
    source: str
    row: int
    source_sha256: str
    raw_amount: str = ""

    @property
    def action_id(self) -> str:
        return _action_id(self.source, self.row)


def split_from_text(
    ex_date: date,
    split_from: str,
    split_to: str,
    source: str,
    row: int,
    source_sha256: str,
) -> SplitInput:
    return SplitInput(
        ex_date,
        parse_decimal_or_none(split_from),
        parse_decimal_or_none(split_to),
        source,
        row,
        source_sha256,
        (split_from, split_to),
    )


def dividend_from_text(
    ex_date: date,
    cash_amount: str,
    dividend_type: str,
    declaration_date: str,
    masked_after_2019: str,
    source: str,
    row: int,
    source_sha256: str,
    currency: str | None = None,
    share_unit: str = PER_SHARE,
) -> DividendInput:
    """Massive 分红行的原文 → DividendInput。masked_after_2019 为以 | 分隔的字段名。"""
    masked = "declaration_date" in {x for x in masked_after_2019.split("|") if x}
    return DividendInput(
        ex_date=ex_date,
        cash_amount=parse_decimal_or_none(cash_amount),
        dividend_type=dividend_type.strip(),
        declaration_date=date.fromisoformat(declaration_date) if declaration_date else None,
        declaration_masked=masked,
        currency=currency,
        share_unit=share_unit,
        source=source,
        row=row,
        source_sha256=source_sha256,
        raw_amount=cash_amount,
    )


# ---- 核实登记（第 5 节） ----


class VerificationKind(StrEnum):
    SAME_DAY_BASIS_POST = "SAME_DAY_BASIS_POST"
    SAME_DAY_BASIS_PRE = "SAME_DAY_BASIS_PRE"
    ABNORMAL_CONFIRMED = "ABNORMAL_CONFIRMED"
    DECL_RECORD_CORRECTED = "DECL_RECORD_CORRECTED"


class TargetType(StrEnum):
    ACTION = "action"
    GROUP = "group"


# 各类登记适用的目标类型：公告日更正针对原始行动，其余针对计算组
KIND_TARGET = {
    VerificationKind.SAME_DAY_BASIS_POST: TargetType.GROUP,
    VerificationKind.SAME_DAY_BASIS_PRE: TargetType.GROUP,
    VerificationKind.ABNORMAL_CONFIRMED: TargetType.GROUP,
    VerificationKind.DECL_RECORD_CORRECTED: TargetType.ACTION,
}

VERIFICATION_COLUMNS = (
    "id",
    "target_type",
    "target",
    "source_sha256",
    "kind",
    "value",
    "ruling",
    "evidence",
)


@dataclass(frozen=True, slots=True)
class Verification:
    id: str
    target_type: TargetType
    target: str  # action_id 或 group_id
    source_sha256: str | None  # target_type = action 时必填
    kind: VerificationKind
    corrected_declaration: date | None  # DECL_RECORD_CORRECTED 的更正后公告日
    confirmed_amount: Decimal | None  # ABNORMAL_CONFIRMED 的核实金额
    ruling: str
    evidence: str

    @property
    def security(self) -> str | None:
        """group 目标的证券（group_id 的第一段）；action 目标由来源判断，这里为 None。"""
        if self.target_type is TargetType.GROUP:
            return self.target.split("|", 1)[0]
        return None

    @property
    def action_source(self) -> str | None:
        if self.target_type is TargetType.ACTION:
            return self.target.rsplit("#", 1)[0]
        return None


def _group_date(group_id: str) -> date | None:
    parts = group_id.split("|")
    if len(parts) < 3:
        return None
    try:
        return date.fromisoformat(parts[2])
    except ValueError:
        return None


def parse_verification(row: dict[str, str]) -> Verification:
    """一行登记的文本 → Verification。先判日期（封存越界报 SealViolation），再解析数值。"""
    vid = row.get("id", "").strip()
    what = f"核实登记 {vid or '(无编号)'}"
    try:
        target_type = TargetType(row.get("target_type", "").strip())
    except ValueError as e:
        raise AdjustmentInputError(f"{what}: target_type 必须为 action 或 group") from e
    try:
        kind = VerificationKind(row.get("kind", "").strip())
    except ValueError as e:
        raise AdjustmentInputError(f"{what}: 未知的 kind {row.get('kind')!r}") from e
    target = row.get("target", "").strip()
    value = row.get("value", "").strip()
    # ① 日期：组身份中的原始除权日、更正后的公告日
    corrected: date | None = None
    if target_type is TargetType.GROUP:
        assert_dates_not_sealed(what, _group_date(target))
    if kind is VerificationKind.DECL_RECORD_CORRECTED:
        try:
            corrected = date.fromisoformat(value)
        except ValueError as e:
            raise AdjustmentInputError(f"{what}: 更正后的公告日不是日期: {value!r}") from e
        assert_dates_not_sealed(what, corrected)
    # ② 其余字段与数值
    if not vid:
        raise AdjustmentInputError("核实登记的 id 不能为空")
    if not target:
        raise AdjustmentInputError(f"{what}: target 不能为空")
    if not row.get("ruling", "").strip() or not row.get("evidence", "").strip():
        raise AdjustmentInputError(f"{what}: ruling 与 evidence 都不能为空")
    if KIND_TARGET[kind] is not target_type:
        raise AdjustmentInputError(f"{what}: {kind} 只适用于 {KIND_TARGET[kind]} 目标")
    sha = row.get("source_sha256", "").strip() or None
    if target_type is TargetType.ACTION and sha is None:
        raise AdjustmentInputError(f"{what}: action 目标必须给出 source_sha256")
    if target_type is TargetType.GROUP and sha is not None:
        raise AdjustmentInputError(f"{what}: group 目标的 source_sha256 必须为空")
    amount: Decimal | None = None
    if kind is VerificationKind.ABNORMAL_CONFIRMED:
        amount = parse_decimal_or_none(value)
        if amount is None:
            raise AdjustmentInputError(f"{what}: 核实金额不是数值: {value!r}")
    elif kind is not VerificationKind.DECL_RECORD_CORRECTED and value:
        raise AdjustmentInputError(f"{what}: {kind} 的 value 必须为空")
    return Verification(
        vid,
        target_type,
        target,
        sha,
        kind,
        corrected,
        amount,
        row["ruling"].strip(),
        row["evidence"].strip(),
    )


def check_verification_ids(verifications: tuple[Verification, ...]) -> None:
    seen: set[str] = set()
    for v in verifications:
        if v.id in seen:
            raise AdjustmentInputError(f"核实登记编号重复: {v.id}")
        seen.add(v.id)
