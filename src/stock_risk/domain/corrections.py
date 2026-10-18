"""复权价更正（I11，T00a）：数据源复权价有已证实的错误时，读取时按登记的方法更正，原始文件不改。

每条更正必须有产品经理裁定编号与数据依据。目前只有一种方法：
- use_raw：区间内每条日线的 adj_open / adj_high / adj_low / adj_close 用对应的原始价替代。
  只适用于区间内没有拆股和分红的情况。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
from enum import StrEnum

from stock_risk.domain.listing import ListingPeriod
from stock_risk.domain.market_data import DailyBar, PriceHistory
from stock_risk.domain.numeric import PARSE_CONTEXT
from stock_risk.domain.raw_text import RawBarText
from stock_risk.domain.seal import assert_dates_not_sealed


class CorrectionError(ValueError):
    """更正登记不合规，或不能应用到给定数据上。"""


class CorrectionMethod(StrEnum):
    USE_RAW = "use_raw"


def parse_method(text: str) -> CorrectionMethod:
    try:
        return CorrectionMethod(text)
    except ValueError as e:
        raise CorrectionError(f"未知的更正方法: {text!r}") from e


@dataclass(frozen=True, slots=True)
class PriceCorrection:
    """一条复权价更正，区间为闭区间 [start, end]。"""

    symbol: str
    start: date
    end: date
    method: CorrectionMethod
    ruling: str  # 产品经理裁定编号，如 PM-03
    evidence: str  # 数据依据
    # T11-D0a 第 6 节：D0b 复核后的标注；absorbed = 由 D0a 吸收。旧行为空（None）
    d0a_status: str | None = None

    def __post_init__(self) -> None:
        if not self.symbol:
            raise CorrectionError("更正的 symbol 不能为空")
        if self.start > self.end:
            raise CorrectionError(f"{self.symbol}: 更正起始日晚于结束日")
        if not isinstance(self.method, CorrectionMethod):
            raise CorrectionError(f"{self.symbol}: 未知的更正方法 {self.method!r}")
        if not self.ruling or not self.evidence:
            raise CorrectionError(f"{self.symbol}: 更正必须有裁定编号与数据依据")
        assert_dates_not_sealed(f"{self.symbol} 更正", self.start, self.end)

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end


def validate_correction(
    correction: PriceCorrection, history: PriceHistory, listing: ListingPeriod | None
) -> None:
    """更正区间必须落在上市期间内；use_raw 的区间内不得有公司行动。"""
    c = correction
    if listing is not None and not (listing.contains(c.start) and listing.contains(c.end)):
        raise CorrectionError(f"{c.symbol}: 更正区间 {c.start} ~ {c.end} 超出上市期间")
    if c.method is CorrectionMethod.USE_RAW:
        inside = [a for a in history.actions if c.contains(a.day)]
        if inside:
            days = ", ".join(f"{a.day} {a.type}" for a in inside)
            raise CorrectionError(f"{c.symbol}: use_raw 区间内有公司行动（{days}）")


def _use_raw(b: DailyBar) -> DailyBar:
    return replace(b, adj_open=b.open, adj_high=b.high, adj_low=b.low, adj_close=b.close)


def apply_corrections(
    history: PriceHistory,
    corrections: Sequence[PriceCorrection],
    listing: ListingPeriod | None = None,
) -> PriceHistory:
    """先逐条校验，再按登记顺序应用。区间外的日线与全部公司行动保持不变。"""
    for c in corrections:
        validate_correction(c, history, listing)
    bars = list(history.bars)
    for c in corrections:
        if c.method is CorrectionMethod.USE_RAW:
            bars = [_use_raw(b) if c.contains(b.day) else b for b in bars]
        else:  # pragma: no cover - 新增方法时必须在此实现
            raise CorrectionError(f"更正方法未实现: {c.method}")
    return PriceHistory(tuple(bars), history.actions)


# ---- 规范数据（T11-D0a 第 6 节第 2 项） ----

D0A_ABSORBED = "absorbed"


def check_canonical_corrections(symbol: str, corrections: Sequence[PriceCorrection]) -> None:
    """读取规范数据时：use_raw 不作用于 F；有未标注 absorbed 的 use_raw 条目即报错。"""
    pending = [
        c
        for c in corrections
        if c.method is CorrectionMethod.USE_RAW and c.d0a_status != D0A_ABSORBED
    ]
    if pending:
        spans = "，".join(f"{c.start} ~ {c.end}（{c.ruling}）" for c in pending)
        raise CorrectionError(
            f"{symbol}: 规范数据下 use_raw 条目须经 D0b 复核并标注 d0a_status = absorbed：{spans}"
        )


# ---- 原始价错误的更正（T11-D0a 第 6 节第 1 项；规格第 9.1 节） ----

RAW_CORRECTION_FIELDS = ("open", "high", "low", "close", "volume")
RAW_CORRECTION_COLUMNS = (
    "id",
    "security",
    "day",
    "field",
    "value",
    "ruling",
    "evidence",
    "version",
)


@dataclass(frozen=True, slots=True)
class RawPriceCorrection:
    """原始价本身有误时的替换值（原文），在复权之前作用于原始文本。不沿用 use_raw 的身份。"""

    id: str
    security: str
    day: date
    field: str
    value: str
    ruling: str
    evidence: str
    version: str


def parse_raw_price_correction(row: dict[str, str]) -> RawPriceCorrection:
    """先判日期（封存越界报 SealViolation），再检查其余字段与数值。"""
    cid = row.get("id", "").strip()
    what = f"原始价更正 {cid or '(无编号)'}"
    try:
        day = date.fromisoformat(row.get("day", "").strip())
    except ValueError as e:
        raise CorrectionError(f"{what}: day 不是日期: {row.get('day')!r}") from e
    assert_dates_not_sealed(what, day)
    field = row.get("field", "").strip()
    value = row.get("value", "").strip()
    if not cid:
        raise CorrectionError("原始价更正的 id 不能为空")
    if not row.get("security", "").strip():
        raise CorrectionError(f"{what}: security 不能为空")
    if field not in RAW_CORRECTION_FIELDS:
        raise CorrectionError(f"{what}: field 必须为 {'/'.join(RAW_CORRECTION_FIELDS)} 之一")
    for name in ("ruling", "evidence", "version"):
        if not row.get(name, "").strip():
            raise CorrectionError(f"{what}: {name} 不能为空")
    with localcontext(PARSE_CONTEXT):
        try:
            number = Decimal(value)
        except InvalidOperation as e:
            raise CorrectionError(f"{what}: 替换值不是数值: {value!r}") from e
    if not number.is_finite() or number < 0:
        raise CorrectionError(f"{what}: 替换值必须是非负有限数: {value!r}")
    return RawPriceCorrection(
        cid,
        row["security"].strip().upper(),
        day,
        field,
        value,
        row["ruling"].strip(),
        row["evidence"].strip(),
        row["version"].strip(),
    )


def apply_raw_price_corrections(
    security: str, bars: Sequence[RawBarText], corrections: Sequence[RawPriceCorrection]
) -> tuple[tuple[RawBarText, ...], tuple[str, ...]]:
    """把属于 security 的更正作用于原始文本；返回 (更正后的文本行, 所用条目编号)。

    同一日同一字段有多条、或对不上任何日线的，报 CorrectionError。"""
    own = [c for c in corrections if c.security == security]
    seen: set[tuple[date, str]] = set()
    for c in own:
        if (c.day, c.field) in seen:
            raise CorrectionError(f"{security} {c.day} {c.field}: 原始价更正重复")
        seen.add((c.day, c.field))
    by_day: dict[date, list[RawPriceCorrection]] = {}
    for c in own:
        by_day.setdefault(c.day, []).append(c)
    days = {b.day for b in bars}
    missing = sorted(d for d in by_day if d not in days)
    if missing:
        raise CorrectionError(f"{security}: 原始价更正对不上任何日线: {missing[0]}")
    out = []
    for b in bars:
        for c in by_day.get(b.day, ()):
            b = replace(b, **{c.field: c.value})
        out.append(b)
    return tuple(out), tuple(sorted(c.id for c in own))
