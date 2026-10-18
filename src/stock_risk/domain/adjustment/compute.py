"""规范前向复权 F（T11-D0a 第 4 节；规格 v4 第 1–7 节；裁决 A-1 至 A-6）。纯函数，不做 I/O。

B_i = ∏ { b ：锚日 a < e′ ≤ i 的已作用计算组 }，F_i = R_i ÷ B_i；F_V,i = V_i × ∏（只乘拆股因子）。
行动先映射到生效有效日 e′（原始除权日当日或之后第一根日线）；e′ 不晚于锚日的为锚日行动，不作用。
拆股与分红的先后按原始除权日判断（规格第 5.1 节）。

数值：因子与 F 都在 PARSE_CONTEXT（28 位，HALF_EVEN）中计算，中间不另行定点量化；
F、F_V 最后一步量化到 6 位小数。
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext

from stock_risk.domain.adjustment.inputs import (
    NONE_CURRENCY,
    PER_SHARE,
    AdjustmentInputError,
    DividendInput,
    SplitInput,
    TargetType,
    Verification,
    VerificationKind,
    check_verification_ids,
)
from stock_risk.domain.adjustment.model import (
    ASSUMPTION_CODES,
    ActionKind,
    ActionRecord,
    AdjustedBar,
    AdjustmentResult,
    Assumption,
    Code,
    Eligibility,
    GroupRecord,
    GroupStatus,
    Issue,
    ReferenceFactor,
)
from stock_risk.domain.adjustment.params import AdjustmentParams
from stock_risk.domain.corrections import RawPriceCorrection, apply_raw_price_corrections
from stock_risk.domain.indicators import true_range
from stock_risk.domain.lifecycle.history_segments import split_segments
from stock_risk.domain.market_data import DailyBar
from stock_risk.domain.numeric import PARSE_CONTEXT, PRICE_QUANTUM, InvalidInputError, _decimal
from stock_risk.domain.observation import DayPresence
from stock_risk.domain.raw_text import RawBarText
from stock_risk.domain.seal import assert_no_sealed_days

ACCEPTED_DIVIDEND_TYPES = ("CD", "SC")  # SC 按裁决 A-2 计入
USD = "USD"
_ONE = Decimal(1)
_PRICE_FIELDS = ("open", "high", "low", "close")
_SAME_DAY_KINDS = (VerificationKind.SAME_DAY_BASIS_POST, VerificationKind.SAME_DAY_BASIS_PRE)


# ---- 原始日线 ----


@dataclass(frozen=True, slots=True)
class _RawBar:
    day: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


def _price(security: str, row: RawBarText, name: str) -> Decimal:
    """原始价：十进制解析，HALF_EVEN 量化到 6 位，量化后须为正（与 quantize_price 同一规则）。"""
    text = getattr(row, name)
    what = f"{security} {row.day} {name}"
    try:
        value = _decimal(text, what).quantize(PRICE_QUANTUM, rounding=ROUND_HALF_EVEN)
    except InvalidOperation as e:
        raise InvalidInputError(f"{what} 无法量化到六位小数: {text!r}") from e
    if value <= 0:
        raise InvalidInputError(f"{what} 量化后不是正数: {text!r}")
    return value


def _parse_raw(security: str, row: RawBarText) -> _RawBar:
    with localcontext(PARSE_CONTEXT):
        o, h, lo, c = (_price(security, row, n) for n in _PRICE_FIELDS)
        v = _decimal(row.volume, f"{security} {row.day} volume")
    if v < 0:
        raise InvalidInputError(f"{security} {row.day} volume 为负数: {row.volume!r}")
    if h < lo:
        raise InvalidInputError(f"{security} {row.day} 原始价: 高 < 低（H={h} L={lo}）")
    if not lo <= c <= h:
        raise InvalidInputError(f"{security} {row.day} 原始价: 收盘不在 [低, 高]")
    return _RawBar(row.day, o, h, lo, c, v)


def _q6(x: Decimal) -> Decimal | None:
    """量化到 6 位小数；数值过大、无法在 28 位内表示的返回 None。"""
    with localcontext(PARSE_CONTEXT):
        try:
            return x.quantize(PRICE_QUANTUM, rounding=ROUND_HALF_EVEN)
        except InvalidOperation:
            return None


# ---- 核实登记 ----


@dataclass(slots=True)
class _Verifs:
    by_action: dict[str, Verification] = field(default_factory=dict)
    by_group: dict[str, list[Verification]] = field(default_factory=dict)
    unrelated: list[str] = field(default_factory=list)
    used: set[str] = field(default_factory=set)

    def group(self, group_id: str, kind: VerificationKind) -> Verification | None:
        for v in self.by_group.get(group_id, ()):
            if v.kind is kind:
                return v
        return None


def split_group_id(security: str, s: SplitInput) -> str:
    return f"{security}|SPLIT|{s.ex_date.isoformat()}|{s.row}"


def dividend_group_id(security: str, d: DividendInput) -> str:
    return f"{security}|DIV|{d.ex_date.isoformat()}|{d.share_unit}|{d.currency or NONE_CURRENCY}"


def _sum(values: Sequence[Decimal | None]) -> Decimal | None:
    if any(v is None for v in values):
        return None
    with localcontext(PARSE_CONTEXT):
        total = Decimal(0)
        for v in values:
            total += v  # type: ignore[operator]
        return total


def _resolve_verifications(
    security: str,
    splits: Sequence[SplitInput],
    dividends: Sequence[DividendInput],
    verifications: Sequence[Verification],
) -> _Verifs:
    """目标属于本证券的登记：对不上、对上多条、哈希不符、类型不适用一律报错（第 5 节）。"""
    check_verification_ids(tuple(verifications))
    inputs: list[SplitInput | DividendInput] = [*splits, *dividends]
    sources = {x.source for x in inputs}
    div_groups: dict[str, list[DividendInput]] = {}
    for d in dividends:
        div_groups.setdefault(dividend_group_id(security, d), []).append(d)
    split_groups = {split_group_id(security, s) for s in splits}
    split_days = {s.ex_date for s in splits}
    out = _Verifs()
    for v in verifications:
        what = f"核实登记 {v.id}（{v.target}）"
        if v.target_type is TargetType.ACTION:
            if v.action_source not in sources:
                out.unrelated.append(v.id)
                continue
            matches = [x for x in inputs if x.action_id == v.target]
            if not matches:
                raise AdjustmentInputError(f"{what}: 对不上任何原始行动")
            if len(matches) > 1:
                raise AdjustmentInputError(f"{what}: 对上多条原始行动")
            target = matches[0]
            if v.source_sha256 != target.source_sha256:
                raise AdjustmentInputError(f"{what}: source_sha256 与当前来源文件不一致")
            if not isinstance(target, DividendInput):
                raise AdjustmentInputError(f"{what}: {v.kind} 不适用于拆股")
            if v.target in out.by_action:
                raise AdjustmentInputError(f"{what}: 同一原始行动有多条公告日更正")
            out.by_action[v.target] = v
            continue
        if v.security != security:
            out.unrelated.append(v.id)
            continue
        if v.target in split_groups:
            raise AdjustmentInputError(f"{what}: {v.kind} 不适用于拆股组")
        members = div_groups.get(v.target)
        if not members:
            raise AdjustmentInputError(f"{what}: 对不上任何计算组")
        existing = out.by_group.get(v.target, [])
        if any(x.kind is v.kind for x in existing):
            raise AdjustmentInputError(f"{what}: 同一计算组有多条 {v.kind} 登记")
        if v.kind in _SAME_DAY_KINDS:
            if members[0].ex_date not in split_days:
                raise AdjustmentInputError(f"{what}: 该组除权日没有拆股，同日口径登记不适用")
            if any(x.kind in _SAME_DAY_KINDS for x in existing):
                raise AdjustmentInputError(f"{what}: 同一计算组的同日口径登记互相冲突")
        if v.kind is VerificationKind.ABNORMAL_CONFIRMED:
            total = _sum([m.cash_amount for m in members])
            if total is None or v.confirmed_amount != total:
                raise AdjustmentInputError(
                    f"{what}: 登记金额 {v.confirmed_amount} 与输入金额 {total} 不一致"
                )
        existing.append(v)
        out.by_group[v.target] = existing
    return out


# ---- 原始行动表 ----


@dataclass(slots=True)
class _Item:
    """一条原始行动在计算中的状态。"""

    record: ActionRecord
    kind: ActionKind
    day: date  # 原始除权日
    group_id: str | None
    effective: date | None  # 记录本身的 e′，只用于判断"终点待处理"
    split_b: Decimal | None = None
    amount: Decimal | None = None


def _split_item(security: str, s: SplitInput, anchor: date, effective: date | None) -> _Item:
    valid = (
        s.split_from is not None and s.split_to is not None and s.split_from > 0 and s.split_to > 0
    )
    b: Decimal | None = None
    if valid:
        with localcontext(PARSE_CONTEXT):
            b = s.split_from / s.split_to  # type: ignore[operator]
    codes: list[str] = []
    gid: str | None = split_group_id(security, s)
    if s.ex_date > anchor:
        codes.append(Code.DECL_DATE_MISSING)  # 数据中没有拆股公告日（裁决 A-1）
        if not valid:
            codes.append(Code.SPLIT_RATIO_INVALID)
            gid = None
    record = ActionRecord(
        action_id=s.action_id,
        security=security,
        action_kind=ActionKind.SPLIT,
        source=s.source,
        row=s.row,
        source_sha256=s.source_sha256,
        original_ex_date=s.ex_date,
        values=s.raw_values,
        dividend_type=None,
        declaration_date=None,
        currency=None,
        group_id=gid,
        codes=tuple(sorted(str(c) for c in codes)),
        verification_ids=(),
        corrected_declaration_date=None,
    )
    return _Item(record, ActionKind.SPLIT, s.ex_date, gid, effective, split_b=b)


def _dividend_item(
    security: str, d: DividendInput, anchor: date, effective: date | None, verifs: _Verifs
) -> _Item:
    codes: list[str] = []
    vids: tuple[str, ...] = ()
    corrected: date | None = None
    gid: str | None = dividend_group_id(security, d)
    if d.ex_date > anchor:
        rejected = False
        if d.dividend_type not in ACCEPTED_DIVIDEND_TYPES:
            codes.append(Code.DIV_TYPE_UNSUPPORTED)
            rejected = True
        if d.cash_amount is None or d.cash_amount <= 0:
            codes.append(Code.DIV_AMOUNT_INVALID)
            rejected = True
        if d.currency is None:
            codes.append(Code.CURRENCY_ASSUMED_USD)  # 裁决 A-3
        elif d.currency != USD:
            codes.append(Code.CURRENCY_NOT_USD)
            rejected = True
        if d.share_unit != PER_SHARE:
            codes.append(Code.UNIT_CONVERSION_FAILED)
            rejected = True
        # 可得性（规格第 3 节，裁决 A-1）；公告日更正后用更正日重新判定（T11-05）
        v = verifs.by_action.get(d.action_id)
        if v is not None:
            corrected = v.corrected_declaration
            vids = (v.id,)
            verifs.used.add(v.id)
            if corrected is None or corrected >= d.ex_date:
                codes.append(Code.DECL_NOT_PRIOR)
        elif d.declaration_masked:  # 公告日晚于 2019-12-31，因而晚于除权日（第 3.3 节）
            codes.append(Code.DECL_NOT_PRIOR)
        elif d.declaration_date is None:
            codes.append(Code.DECL_DATE_MISSING)
        elif d.declaration_date >= d.ex_date:
            codes.append(Code.DECL_NOT_PRIOR)
        if rejected:
            gid = None
    record = ActionRecord(
        action_id=d.action_id,
        security=security,
        action_kind=ActionKind.DIVIDEND,
        source=d.source,
        row=d.row,
        source_sha256=d.source_sha256,
        original_ex_date=d.ex_date,
        values=(d.raw_amount,),
        dividend_type=d.dividend_type,
        declaration_date=d.declaration_date,
        currency=d.currency,
        group_id=gid,
        codes=tuple(sorted(str(c) for c in codes)),
        verification_ids=vids,
        corrected_declaration_date=corrected,
    )
    return _Item(record, ActionKind.DIVIDEND, d.ex_date, gid, effective, amount=d.cash_amount)


# ---- 计算组 ----


@dataclass(slots=True)
class _Group:
    group_id: str
    kind: ActionKind
    day: date
    items: list[_Item]
    effective: date | None
    status: GroupStatus = GroupStatus.PENDING
    b: Decimal | None = None
    p: date | None = None
    r_p: Decimal | None = None
    k: int | None = None
    total: Decimal | None = None
    factors: tuple[ReferenceFactor, ...] = ()
    c_ref: Decimal | None = None
    theoretical: bool | None = None
    codes: list[str] = field(default_factory=list)
    vids: list[str] = field(default_factory=list)

    @property
    def members(self) -> tuple[str, ...]:
        return tuple(sorted(i.record.action_id for i in self.items))

    def record(self, security: str) -> GroupRecord:
        if self.status is GroupStatus.PENDING:  # 只保留身份、成员与原始除权日（第 4.8 节）
            return GroupRecord(
                self.group_id, security, self.kind, self.members, self.day, None, self.status
            )
        return GroupRecord(
            group_id=self.group_id,
            security=security,
            action_kind=self.kind,
            members=self.members,
            original_ex_date=self.day,
            effective_day=self.effective,
            status=self.status,
            b=self.b,
            p=self.p,
            r_p=self.r_p,
            k=self.k,
            dividend_total=self.total,
            reference_factors=self.factors,
            c_ref=self.c_ref,
            reference_is_theoretical=self.theoretical,
            codes=tuple(sorted({str(c) for c in self.codes})),
            verification_ids=tuple(sorted(self.vids)),
        )


def _order(g: _Group) -> tuple[date, int, str]:
    return (g.day, 0 if g.kind is ActionKind.SPLIT else 1, g.group_id)


@dataclass(frozen=True, slots=True)
class _Reference:
    """一个 e′ 上的参考日 p、原始收盘 R_p、缺价交易日数 k 与跨历史段的 G。"""

    p: date
    r_p: Decimal
    k: int
    cross_segment_gaps: tuple[int, ...]


def _process_dividend(
    g: _Group,
    ref: _Reference,
    earlier: Sequence[_Group],
    loose: Sequence[_Item],
    same_day_splits: Sequence[_Item],
    same_day_div_groups: int,
    verifs: _Verifs,
    params: AdjustmentParams,
) -> None:
    """一组分红（规格第 5.1–5.4 节；裁决 A-2、A-4、A-5）。

    earlier：同一 e′ 上原始除权日 ∈ (p, d) 的组（已处理）；loose：被拒绝而没有组的行动。"""
    g.p, g.r_p, g.k = ref.p, ref.r_p, ref.k
    g.total = _sum([i.amount for i in g.items])
    reject: list[str] = []
    if same_day_div_groups > 1:  # 同一原始除权日、单位或币种不同的分红不能直接相加
        reject.append(Code.UNIT_CONVERSION_FAILED)
    basis: VerificationKind | None = None
    split_b: Decimal | None = None
    if same_day_splits:
        if len(same_day_splits) > 1:  # 金额对应哪一阶段的单位不明，拒绝
            reject.append(Code.SAME_DAY_MULTI_SPLIT)
        elif same_day_splits[0].split_b is None:
            reject.append(Code.UNIT_CONVERSION_FAILED)
        else:
            split_b = same_day_splits[0].split_b
            for kind in _SAME_DAY_KINDS:
                v = verifs.group(g.group_id, kind)
                if v is not None:
                    basis = kind
                    g.vids.append(v.id)
                    verifs.used.add(v.id)
            if basis is None:
                reject.append(Code.SAME_DAY_BASIS_UNVERIFIED)
    if ref.cross_segment_gaps:  # 先判这一条（第 4.5 节）
        reject.append(Code.NO_REFERENCE_CLOSE)
    elif ref.k > params.fallback_max_missing_days:
        reject.append(Code.FALLBACK_TOO_FAR)
    factors: list[ReferenceFactor] = []
    for e in earlier:
        if e.status is GroupStatus.APPLIED and e.b is not None:
            factors.append(ReferenceFactor(e.group_id, e.b))
        else:
            reject.append(Code.UNIT_CONVERSION_FAILED)
    if any(ref.p < i.day < g.day for i in loose):  # 期间有无法换算的行动
        reject.append(Code.UNIT_CONVERSION_FAILED)
    g.factors = tuple(factors)
    if g.effective != g.day:
        g.codes.append(Code.ACTION_DATE_SHIFTED)
    if reject or g.total is None:
        g.status = GroupStatus.REJECTED
        g.codes.extend(reject)
        return
    with localcontext(PARSE_CONTEXT):
        c_ref = ref.r_p
        for f in factors:
            c_ref = c_ref * f.b
        amount = g.total
        if split_b is not None:
            c_ref = c_ref * split_b  # 换算到拆股后每股
            if basis is VerificationKind.SAME_DAY_BASIS_PRE:
                amount = amount * split_b  # D′ = D × b
        b = _ONE - amount / c_ref
    g.c_ref, g.b = c_ref, b
    # 参考价不是观察值：缺价替代（k > 0）或依次递推的理论价（期间有更早的分红组）
    g.theoretical = ref.k > 0 or any(e.kind is ActionKind.DIVIDEND for e in earlier)
    if g.theoretical:
        g.codes.append(Code.DIV_PREV_FALLBACK)
    if b <= 0:
        g.codes.append(Code.ILLEGAL_FACTOR)  # 任何登记都不能放行（裁决 A-4）
        g.status = GroupStatus.REJECTED
        return
    if b < params.abnormal_dividend_factor:
        v = verifs.group(g.group_id, VerificationKind.ABNORMAL_CONFIRMED)
        if v is None:
            g.codes.append(Code.ABNORMAL_FACTOR)
            g.status = GroupStatus.REJECTED
            return
        g.vids.append(v.id)
        verifs.used.add(v.id)
    g.status = GroupStatus.APPLIED


def _build_groups(security: str, items: Sequence[_Item], bar_days: Sequence[date]) -> list[_Group]:
    by_id: dict[str, _Group] = {}
    for it in items:
        if it.group_id is None:
            continue
        g = by_id.get(it.group_id)
        if g is None:
            i = bisect.bisect_left(bar_days, it.day)
            effective = bar_days[i] if i < len(bar_days) else None
            g = by_id[it.group_id] = _Group(it.group_id, it.kind, it.day, [], effective)
        g.items.append(it)
    return sorted(by_id.values(), key=_order)


def _resolve_groups(
    groups: Sequence[_Group],
    items: Sequence[_Item],
    raw: Sequence[_RawBar],
    calendar: Sequence[date],
    anchor: date,
    verifs: _Verifs,
    params: AdjustmentParams,
) -> None:
    bar_index = {b.day: i for i, b in enumerate(raw)}
    loose = [it for it in items if it.group_id is None and it.day > anchor]
    split_items_by_day: dict[date, list[_Item]] = {}
    for it in items:
        if it.kind is ActionKind.SPLIT:
            split_items_by_day.setdefault(it.day, []).append(it)
    div_groups_by_day: dict[date, int] = {}
    for g in groups:
        if g.kind is ActionKind.DIVIDEND:
            div_groups_by_day[g.day] = div_groups_by_day.get(g.day, 0) + 1
    by_effective: dict[date, list[_Group]] = {}
    for g in groups:
        if g.day <= anchor or (g.effective is not None and g.effective <= anchor):
            g.status = GroupStatus.ANCHOR_NOT_APPLIED  # 规格第 2 节、第 5.5 节
        elif g.effective is None:
            g.status = GroupStatus.PENDING  # 终点待处理：之后再无有效日线
        else:
            by_effective.setdefault(g.effective, []).append(g)
    for e, bucket in sorted(by_effective.items()):
        bucket.sort(key=_order)
        idx = bar_index[e]
        prev = raw[idx - 1]  # e′ 晚于锚日，前面必有日线
        k = bisect.bisect_left(calendar, e) - bisect.bisect_right(calendar, prev.day)
        cross = tuple(gb for gb in params.gap_breaks if k >= gb + 1)
        ref = _Reference(prev.day, prev.close, k, cross)
        for g in bucket:
            if g.kind is ActionKind.SPLIT:
                g.b = g.items[0].split_b
                g.status = GroupStatus.APPLIED
                if g.effective != g.day:
                    g.codes.append(Code.ACTION_DATE_SHIFTED)
        for g in bucket:
            if g.kind is not ActionKind.DIVIDEND:
                continue
            earlier = [x for x in bucket if ref.p < x.day < g.day]
            same_day = [
                s
                for s in split_items_by_day.get(g.day, ())
                if s.day > anchor  # 同日拆股（原始除权日 = d）
            ]
            _process_dividend(
                g,
                ref,
                earlier,
                loose,
                same_day,
                div_groups_by_day.get(g.day, 0),
                verifs,
                params,
            )


# ---- F 与检查 ----


def _adjust(raw: Sequence[_RawBar], groups: Sequence[_Group]) -> tuple[AdjustedBar, ...]:
    applied: dict[date, list[_Group]] = {}
    for g in groups:
        if g.status is GroupStatus.APPLIED and g.effective is not None:
            applied.setdefault(g.effective, []).append(g)
    out: list[AdjustedBar] = []
    b_cum, s_cum = _ONE, _ONE
    with localcontext(PARSE_CONTEXT):
        for r in raw:
            for g in sorted(applied.get(r.day, ()), key=_order):
                assert g.b is not None
                b_cum = b_cum * g.b
                if g.kind is ActionKind.SPLIT:
                    s_cum = s_cum * g.b
            f = [_q6(getattr(r, n) / b_cum) for n in _PRICE_FIELDS]
            out.append(
                AdjustedBar(
                    r.day,
                    r.open,
                    r.high,
                    r.low,
                    r.close,
                    r.volume,
                    b_cum,
                    f[0],
                    f[1],
                    f[2],
                    f[3],
                    _q6(r.volume * s_cum),
                )
            )
    return tuple(out)


def _representability(
    security: str, bars: Sequence[AdjustedBar], params: AdjustmentParams
) -> tuple[list[Issue], list[date]]:
    """四个 F 价都在 [下限, 上限]（含端点），高低开收关系完整成立（第 4.6 节）；振幅坍缩只计数。"""
    issues: list[Issue] = []
    collapse: list[date] = []
    for b in bars:
        prices = {n: getattr(b, f"f_{n}") for n in _PRICE_FIELDS}
        for n, v in prices.items():
            if v is None or not params.price_lower <= v <= params.price_upper:
                issues.append(
                    Issue(
                        security,
                        Code.REPRESENTABILITY.value,
                        b.day,
                        n,
                        "无法量化" if v is None else str(v),
                        f"复权价须在 [{params.price_lower}, {params.price_upper}] 之内",
                        "bar",
                    )
                )
        o, h, lo, c = (prices[n] for n in _PRICE_FIELDS)
        if o is None or h is None or lo is None or c is None:
            continue
        if not (h >= lo and lo <= o <= h and lo <= c <= h):
            issues.append(
                Issue(
                    security,
                    Code.REPRESENTABILITY.value,
                    b.day,
                    "ohlc",
                    f"O={o} H={h} L={lo} C={c}",
                    "高低开收关系不成立",
                    "bar",
                )
            )
        if b.high > b.low and h == lo:
            collapse.append(b.day)
    return issues, collapse


def _float_bar(b: AdjustedBar) -> DailyBar | None:
    if b.f_open is None or b.f_high is None or b.f_low is None or b.f_close is None:
        return None
    return DailyBar(
        b.day,
        float(b.open),
        float(b.high),
        float(b.low),
        float(b.close),
        float(b.volume),
        float(b.f_open),
        float(b.f_high),
        float(b.f_low),
        float(b.f_close),
    )


def _atr_precheck(
    security: str,
    bars: Sequence[AdjustedBar],
    calendar: Sequence[date],
    params: AdjustmentParams,
) -> list[Issue]:
    """每个 G、每个历史段、每个窗口长度：ATR ≤ 0 记 ATR_NONPOSITIVE（第 4.7 节）。

    TR 用 indicators.true_range（同一组 float F 值），ATR = fsum(TR) ÷ n，
    与 indicators.atr 相同。"""
    if not bars:
        return []
    first, last = bars[0].day, bars[-1].day
    by_day = {b.day: b for b in bars}
    lo, hi = bisect.bisect_left(calendar, first), bisect.bisect_right(calendar, last)
    presence = [DayPresence(d, d in by_day) for d in calendar[lo:hi]]
    issues: list[Issue] = []
    for gap in params.gap_breaks:
        for seg in split_segments(presence, gap):
            seg_bars = [_float_bar(by_day[d]) for d in seg.bar_days]
            trs: list[float | None] = [None]
            for j in range(1, len(seg_bars)):
                if seg_bars[j] is None or seg_bars[j - 1] is None:
                    trs.append(None)
                else:
                    trs.append(true_range(seg_bars, j))  # type: ignore[arg-type]
            for n in params.atr_windows:
                for i in range(n, len(seg_bars)):
                    window = trs[i - n + 1 : i + 1]
                    if any(t is None for t in window):
                        continue
                    value = math.fsum(window) / n  # type: ignore[arg-type]
                    if not value > 0:
                        issues.append(
                            Issue(
                                security,
                                Code.ATR_NONPOSITIVE.value,
                                seg.bar_days[i],
                                f"ATR{n}",
                                repr(value),
                                f"G={gap}",
                                "bar",
                            )
                        )
    return issues


# ---- 入口 ----


def compute_canonical(
    security: str,
    bars_text: Sequence[RawBarText],
    splits: Sequence[SplitInput],
    dividends: Sequence[DividendInput],
    calendar_days: Sequence[date],
    verifications: Sequence[Verification],
    raw_corrections: Sequence[RawPriceCorrection],
    params: AdjustmentParams,
) -> AdjustmentResult:
    """由原始日线与公司行动计算规范前向复权（第 4.1 节）。

    bars_text：日期升序，已按现行 segments 拼接；calendar_days：XNYS 交易日，升序。"""
    if not bars_text:
        raise InvalidInputError(f"{security}: 没有日线")
    assert_no_sealed_days((r.day for r in bars_text), f"{security} 日线")
    assert_no_sealed_days((s.ex_date for s in splits), f"{security} 拆股")
    assert_no_sealed_days((d.ex_date for d in dividends), f"{security} 分红")
    assert_no_sealed_days(calendar_days, "交易日历")
    calendar = list(calendar_days)
    if any(a >= b for a, b in zip(calendar, calendar[1:], strict=False)):
        raise InvalidInputError("交易日历未严格升序")
    corrected_text, correction_ids = apply_raw_price_corrections(
        security, bars_text, raw_corrections
    )
    raw = [_parse_raw(security, r) for r in corrected_text]
    days = [r.day for r in raw]
    if any(a >= b for a, b in zip(days, days[1:], strict=False)):
        raise InvalidInputError(f"{security}: 日线日期未严格升序（重复或乱序）")
    cal_set = set(calendar)
    outside = [d for d in days if d not in cal_set]
    if outside:
        raise InvalidInputError(f"{security}: 日线日期不在交易日历中: {outside[0]}")
    anchor = days[0]

    verifs = _resolve_verifications(security, splits, dividends, verifications)

    def effective_of(d: date) -> date | None:
        i = bisect.bisect_left(days, d)
        return days[i] if i < len(days) else None

    items = [_split_item(security, s, anchor, effective_of(s.ex_date)) for s in splits]
    items += [
        _dividend_item(security, d, anchor, effective_of(d.ex_date), verifs) for d in dividends
    ]
    ids = [it.record.action_id for it in items]
    if len(set(ids)) != len(ids):
        raise AdjustmentInputError(f"{security}: 原始行动身份重复（同一来源文件同一行号）")
    groups = _build_groups(security, items, days)
    _resolve_groups(groups, items, raw, calendar, anchor, verifs, params)
    bars = _adjust(raw, groups)

    issues: list[Issue] = []
    assumptions: list[Assumption] = []
    counted: list[Issue] = []  # 计入资格判断的问题（终点待处理的原始记录不计）
    for it in sorted(items, key=lambda x: x.record.action_id):
        rec = it.record
        for code in rec.codes:
            if code in ASSUMPTION_CODES:
                assumptions.append(
                    Assumption(
                        security,
                        code,
                        "action",
                        rec.action_id,
                        it.day,
                        None,
                        rec.source,
                        None,
                        it.day,
                    )
                )
            else:
                issue = Issue(security, code, it.day, "action", rec.action_id, "", "action")
                issues.append(issue)
                if it.effective is not None:
                    counted.append(issue)
    for g in groups:
        if g.effective is None or g.status in (GroupStatus.PENDING, GroupStatus.ANCHOR_NOT_APPLIED):
            continue
        source = ";".join(sorted({i.record.source for i in g.items}))
        for code in sorted({str(c) for c in g.codes}):
            if code in ASSUMPTION_CODES:
                k = g.k if code == Code.DIV_PREV_FALLBACK else None
                assumptions.append(
                    Assumption(
                        security,
                        code,
                        "group",
                        g.group_id,
                        g.day,
                        g.effective,
                        source,
                        k,
                        g.effective,
                    )
                )
            else:
                note = ""
                if code == Code.NO_REFERENCE_CLOSE:
                    gaps = [gb for gb in params.gap_breaks if (g.k or 0) >= gb + 1]
                    note = "G=" + ",".join(str(x) for x in gaps)
                issue = Issue(security, code, g.effective, "group", g.group_id, note, "group")
                issues.append(issue)
                counted.append(issue)
    rep_issues, collapse = _representability(security, bars, params)
    atr_issues = _atr_precheck(security, bars, calendar, params)
    issues += rep_issues + atr_issues
    counted += rep_issues + atr_issues
    issues.sort(key=lambda x: (x.day, x.code, x.field, x.value, x.note))
    assumptions.sort(key=lambda x: (x.decision_day, x.code, x.target_id))
    reasons = tuple(sorted({i.code for i in counted}))
    return AdjustmentResult(
        security=security,
        anchor=anchor,
        bars=bars,
        actions=tuple(sorted((it.record for it in items), key=lambda r: r.action_id)),
        groups=tuple(sorted((g.record(security) for g in groups), key=lambda r: r.group_id)),
        assumptions=tuple(assumptions),
        issues=tuple(issues),
        eligibility=Eligibility.NOT_USABLE if reasons else Eligibility.USABLE,
        reasons=reasons,
        amplitude_collapse_days=tuple(collapse),
        terminal_pending=tuple(
            sorted(g.group_id for g in groups if g.status is GroupStatus.PENDING and g.day > anchor)
        ),
        raw_correction_ids=correction_ids,
        verification_ids_used=tuple(sorted(verifs.used)),
        unrelated_verification_ids=tuple(sorted(verifs.unrelated)),
    )
