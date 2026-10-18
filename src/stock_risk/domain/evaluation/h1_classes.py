"""17 §4「未知的分类、上限与敏感性报告」/ T13-H1C：H1 未知的甲、乙分类。纯函数，不做 I/O。

- 方法状态的完整性（第 1.2 节）：method_state；
- 单个事件的分类（第 1.0 节，"先判定，后分类"）：classify_h1；入口先校验调用契约；
- 原因集合的分类（第 1.1 节的表）：classify_reasons；依赖未决的核实：verify_dependency；
- 区间单位：interval_unit；单位身份、预期单位清单与预期方法名单的检查（第 1.3 节）：
  expected_units、check_units；
- 汇总：应检单位、四个比例、敏感性与上限判定（第 2 节）：summarize；
- B3（无逐日状态的方法；17 号第 4 节 B3 澄清）：只参加覆盖与甲类集合的跨方法一致检查，
  汇总函数直接拒绝。

U_甲、U_退市尚未批准：本模块不设任何默认值，只能由调用方以 Decimal 传入，或显式传"未批准"。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum

from stock_risk.domain.evaluation.decimal_ratio import ratio
from stock_risk.domain.evaluation.event_model import SevereEvent
from stock_risk.domain.evaluation.h1 import FRef, H1Result, H1Status
from stock_risk.domain.evaluation.method_series import MethodSeries
from stock_risk.domain.evaluation.outcomes import exclusion_reasons


class H1ContractError(ValueError):
    """调用契约或检查不成立：按工程错误处理（17 号第 5.4 节），不归入任何类别。"""


class MethodState(StrEnum):
    COMPLETE = "COMPLETE"
    NOT_RUN = "NOT_RUN"
    FAILED = "FAILED"
    MISALIGNED = "MISALIGNED"  # 缺日、日期错位或顺序不同


class MethodKind(StrEnum):
    DAILY = "DAILY"  # 有逐日在警状态（R0、B1、B2、候选）
    NO_DAILY_STATE = "NO_DAILY_STATE"  # 没有逐日状态（B3）


class UnitKind(StrEnum):
    EVENT = "EVENT"
    INTERVAL = "INTERVAL"


class UnitResult(StrEnum):
    OK = "OK"
    VIOLATION = "VIOLATION"
    CLASS_B = "CLASS_B"  # 乙类：计为违反
    CLASS_A = "CLASS_A"  # 甲类：单列，计入比例
    NOT_JUDGED = "NOT_JUDGED"  # 主口径事件，F_ref 已知不存在；不计入应检单位
    NOT_APPLICABLE_NO_DAILY_STATE = "NOT_APPLICABLE_NO_DAILY_STATE"  # 只用于 B3


class Category(StrEnum):
    DELISTED = "DELISTED"  # 退市
    MISSING = "MISSING"  # 缺价
    WARMUP = "WARMUP"  # 预热
    RIGHT_CENSORED = "RIGHT_CENSORED"  # 右截尾


class ClassBReason(StrEnum):
    METHOD_STATE_UNKNOWN = "METHOD_STATE_UNKNOWN"
    UNRECOGNIZED_REASON = "UNRECOGNIZED_REASON"
    DEPENDENCY_UNVERIFIED = "DEPENDENCY_UNVERIFIED"


# ---- 第 1.1 节的表：只认逐字列出的代码 ----

RIGHT_CODES: Mapping[str, Category] = {
    "RIGHT_DELISTED": Category.DELISTED,
    "RIGHT_DATA_END": Category.RIGHT_CENSORED,
    "RIGHT_REQUIRED_END": Category.RIGHT_CENSORED,
    "RIGHT_PERIOD_END": Category.RIGHT_CENSORED,
}
DIRECT_CODES: Mapping[str, Category] = {
    **RIGHT_CODES,
    "MISSING": Category.MISSING,
    "DATA_BREAK": Category.MISSING,
    "UNEVALUABLE_MISSING": Category.MISSING,
    "UNEVALUABLE_WARMUP": Category.WARMUP,  # 预热已由 events._unevaluable 在事件生成时证明
}
F_REF_ABSENT = "F_REF_ABSENT"
DEPENDENCY_PENDING = "DEPENDENCY_PENDING"


@dataclass(frozen=True, slots=True)
class ReasonClassification:
    class_a: bool
    categories: frozenset[Category]  # 只有甲类有
    unrecognized: frozenset[str]
    dependency_unverified: bool


def classify_reasons(
    codes: frozenset[str], dependency_category: Category | None = None
) -> ReasonClassification:
    """原因集合的分类（第 1.1 节）。dependency_category：依赖已核实时它的类别，核实不了为 None。

    原因集合非空、且每个代码都满足条件 → 甲类；否则乙类（未识别的代码全部列出）。"""
    cats: set[Category] = set()
    unrecognized: set[str] = set()
    dependency_unverified = False
    rights = [c for c in codes if c in RIGHT_CODES]
    for code in codes:
        if code in DIRECT_CODES:
            cats.add(DIRECT_CODES[code])
        elif code == F_REF_ABSENT:
            if rights:  # 跟随同一集合中的 RIGHT_* 代码
                cats.add(
                    Category.DELISTED if "RIGHT_DELISTED" in codes else Category.RIGHT_CENSORED
                )
            else:
                unrecognized.add(code)
        elif code == DEPENDENCY_PENDING:
            if dependency_category is None:
                dependency_unverified = True
            else:
                cats.add(dependency_category)
        else:
            unrecognized.add(code)
    class_a = bool(codes) and not unrecognized and not dependency_unverified
    return ReasonClassification(
        class_a,
        frozenset(cats) if class_a else frozenset(),
        frozenset(unrecognized),
        dependency_unverified,
    )


def verify_dependency(event: SevereEvent, data_end: date) -> Category | None:
    """DEPENDENCY_PENDING 的五项核实条件（第 1.3 节）；全部成立时返回类别，否则 None。"""
    dep = event.dependency_pending
    if dep is None:  # 1
        return None
    if event.confirm is not None or event.e is None:  # 2
        return None
    if dep.start_exclusive != event.e:  # 3
        return None
    if dep.end_inclusive != data_end:  # 4
        return None
    rights = [c for c in event.censor if c.startswith("RIGHT_")]
    if len(rights) != 1 or rights[0] not in RIGHT_CODES:  # 5
        return None
    return RIGHT_CODES[rights[0]]


# ---- 单位 ----


@dataclass(frozen=True, slots=True, order=True)
class UnitId:
    """事件单位：(证券, 事件集, s, p)；区间单位另加 (start_exclusive, end_inclusive)。"""

    security: str
    event_set: str
    s: date
    p: date
    start_exclusive: date | None = None
    end_inclusive: date | None = None

    @property
    def kind(self) -> UnitKind:
        return UnitKind.EVENT if self.start_exclusive is None else UnitKind.INTERVAL


@dataclass(frozen=True, slots=True)
class H1Unit:
    unit_id: UnitId
    kind: UnitKind
    method: str
    result: UnitResult
    codes: frozenset[str]
    categories: frozenset[Category]  # 只有甲类有
    b_reasons: frozenset[ClassBReason]  # 乙类的原因；B3 的不适用单位保留为诊断
    unrecognized: frozenset[str]
    method_state: MethodState | None  # B3 没有逐日状态，为 None
    as_of: date  # 归属日期：事件单位取 p；区间单位取所依赖事件的 p
    valid_days: int | None = None  # 只有区间单位有


def event_unit_id(security: str, event_set: str, event: SevereEvent) -> UnitId:
    return UnitId(security, event_set, event.s, event.p)


def interval_unit_id(security: str, event_set: str, event: SevereEvent) -> UnitId | None:
    """非空的依赖未决区间（start_exclusive < end_inclusive，不按 valid_days 判断）。"""
    dep = event.dependency_pending
    if dep is None or not dep.start_exclusive < dep.end_inclusive:
        return None
    return UnitId(security, event_set, event.s, event.p, dep.start_exclusive, dep.end_inclusive)


def method_state(
    calendar_days: Sequence[date], method: MethodSeries | None, absent: MethodState | None = None
) -> MethodState:
    """第 1.2 节：方法序列的 days 与评价日历逐项相等才是 COMPLETE。没有序列的，按调用方给出的
    NOT_RUN 或 FAILED 处理。不得用"不在警"填补任何不完整的状态。"""
    if method is None:
        if absent not in (MethodState.NOT_RUN, MethodState.FAILED):
            raise H1ContractError("没有方法序列时，必须给出 NOT_RUN 或 FAILED")
        return absent  # type: ignore[return-value]
    if absent is not None:
        raise H1ContractError("给出了方法序列，就不得同时给出 NOT_RUN 或 FAILED")
    return (
        MethodState.COMPLETE
        if tuple(method.days) == tuple(calendar_days)
        else MethodState.MISALIGNED
    )


def _check_contract(event: SevereEvent, f_ref: FRef) -> None:
    """入口契约（G-T13-01、T13R-02），与 find_f_ref 的生成路径一致。"""
    if f_ref.searched != (event.is_main or event.is_prefix_complete):
        raise H1ContractError("F_ref.searched 必须等于（主口径 或 前缀完整）")
    if not f_ref.searched:
        if f_ref.day is not None:
            raise H1ContractError("未搜索的 F_ref 不得有日期")
        if f_ref.reasons != exclusion_reasons(event):
            raise H1ContractError("未搜索的 F_ref 原因必须等于 exclusion_reasons(event)")
    if f_ref.day is not None and not f_ref.searched:  # pragma: no cover - 上面已覆盖
        raise H1ContractError("有 F_ref 日期时必须已搜索")


def classify_h1(
    security: str,
    event_set: str,
    method: str,
    event: SevereEvent,
    f_ref: FRef,
    judged: H1Result | None,
    state: MethodState,
    data_end: date,
) -> H1Unit:
    """逐日方法的事件单位（第 1.0 节）。

    judged：现有 judge_h1 的结果；state 不是 COMPLETE 时必须为空。"""
    _check_contract(event, f_ref)
    if state is not MethodState.COMPLETE and judged is not None:
        raise H1ContractError("方法状态不完整时不得调用 judge_h1")
    uid = event_unit_id(security, event_set, event)

    def unit(
        result: UnitResult,
        codes: frozenset[str] = frozenset(),
        cats: frozenset[Category] = frozenset(),
        b: frozenset[ClassBReason] = frozenset(),
        unrec: frozenset[str] = frozenset(),
    ) -> H1Unit:
        return H1Unit(uid, UnitKind.EVENT, method, result, codes, cats, b, unrec, state, event.p)

    # 第 1 步：能判定的，照常判定
    if f_ref.searched and f_ref.day is not None:
        if state is MethodState.COMPLETE:
            if judged is None or judged.status not in (H1Status.OK, H1Status.VIOLATION):
                raise H1ContractError("F_ref 已找到且方法完整时，judge_h1 应给出 OK 或 VIOLATION")
            return unit(UnitResult(judged.status.value))
        return unit(UnitResult.CLASS_B, b=frozenset({ClassBReason.METHOD_STATE_UNKNOWN}))
    # 第 2 步：已知不适用
    if event.is_main and f_ref.searched and f_ref.day is None:
        return unit(UnitResult.NOT_JUDGED, codes=f_ref.reasons)
    # 第 3 步：其余事件
    return _chain_unit(uid, method, event, f_ref, state, data_end)


def _chain_unit(
    uid: UnitId,
    method: str,
    event: SevereEvent,
    f_ref: FRef,
    state: MethodState | None,
    data_end: date,
) -> H1Unit:
    codes = exclusion_reasons(event) | f_ref.reasons
    dep = verify_dependency(event, data_end) if event.dependency_pending is not None else None
    c = classify_reasons(codes, dep)
    if c.class_a:
        return H1Unit(
            uid, UnitKind.EVENT, method, UnitResult.CLASS_A, codes, c.categories,
            frozenset(), frozenset(), state, event.p,
        )  # fmt: skip
    b: set[ClassBReason] = set()
    if c.dependency_unverified:
        b.add(ClassBReason.DEPENDENCY_UNVERIFIED)
    if c.unrecognized or not codes:
        b.add(ClassBReason.UNRECOGNIZED_REASON)
    return H1Unit(
        uid, UnitKind.EVENT, method, UnitResult.CLASS_B, codes, frozenset(),
        frozenset(b), c.unrecognized, state, event.p,
    )  # fmt: skip


def interval_unit(
    security: str,
    event_set: str,
    method: str,
    event: SevereEvent,
    state: MethodState | None,
    data_end: date,
) -> H1Unit | None:
    """区间单位（第 1.4 节）：核实成立的归甲，类别只由所依赖事件已核实的唯一 RIGHT_* 代码确定；
    否则归乙（DEPENDENCY_UNVERIFIED）。与方法无关。"""
    uid = interval_unit_id(security, event_set, event)
    if uid is None:
        return None
    dep = event.dependency_pending
    assert dep is not None
    cat = verify_dependency(event, data_end)
    codes = frozenset({DEPENDENCY_PENDING})
    if cat is not None:
        return H1Unit(
            uid, UnitKind.INTERVAL, method, UnitResult.CLASS_A, codes, frozenset({cat}),
            frozenset(), frozenset(), state, event.p, dep.valid_days,
        )  # fmt: skip
    return H1Unit(
        uid, UnitKind.INTERVAL, method, UnitResult.CLASS_B, codes, frozenset(),
        frozenset({ClassBReason.DEPENDENCY_UNVERIFIED}), frozenset(), state, event.p,
        dep.valid_days,
    )  # fmt: skip


@dataclass(frozen=True, slots=True)
class EventInput:
    """一个事件及其 F_ref；judged 为现有 judge_h1 的结果（方法不完整或 B3 时为空）。"""

    event: SevereEvent
    f_ref: FRef
    judged: H1Result | None = None


def method_units(
    security: str,
    event_set: str,
    method: str,
    kind: MethodKind,
    state: MethodState | None,
    items: Sequence[EventInput],
    data_end: date,
) -> tuple[H1Unit, ...]:
    """一个方法在一个事件集上的全部单位：每个事件一个事件单位，每个非空依赖区间一个区间单位。

    逐日方法按 classify_h1；B3（无逐日状态）只保留与方法无关的甲类，其余单位一律
    NOT_APPLICABLE_NO_DAILY_STATE，并保留源原因代码与诊断（T13R-03）。"""
    out: list[H1Unit] = []
    for it in items:
        if kind is MethodKind.DAILY:
            if state is None:
                raise H1ContractError("逐日方法必须给出方法状态")
            out.append(
                classify_h1(
                    security, event_set, method, it.event, it.f_ref, it.judged, state, data_end
                )
            )
            iv = interval_unit(security, event_set, method, it.event, state, data_end)
        else:
            if state is not None or it.judged is not None:
                raise H1ContractError("无逐日状态的方法不得有方法状态或 judge_h1 结果")
            _check_contract(it.event, it.f_ref)
            uid = event_unit_id(security, event_set, it.event)
            f = it.f_ref
            if f.searched and (f.day is not None or it.event.is_main):
                # 第 1、2 步的单位：B3 没有逐日状态，不判定，记不适用（不复用逐日方法的结果）
                out.append(
                    H1Unit(
                        uid, UnitKind.EVENT, method, UnitResult.NOT_APPLICABLE_NO_DAILY_STATE,
                        f.reasons, frozenset(), frozenset(), frozenset(), None, it.event.p,
                    )
                )  # fmt: skip
            else:
                out.append(_no_daily(_chain_unit(uid, method, it.event, f, None, data_end)))
            iv = interval_unit(security, event_set, method, it.event, None, data_end)
            iv = _no_daily(iv) if iv is not None else None
        if iv is not None:
            out.append(iv)
    return tuple(out)


def _no_daily(u: H1Unit) -> H1Unit:
    """B3：按事件链归甲的单位照留（与方法无关）；其余改为 NOT_APPLICABLE_NO_DAILY_STATE，
    保留源原因代码与诊断（乙类原因、未识别代码），不计入任何违反数。"""
    if u.result is UnitResult.CLASS_A:
        return u
    return H1Unit(
        u.unit_id, u.kind, u.method, UnitResult.NOT_APPLICABLE_NO_DAILY_STATE, u.codes,
        frozenset(), u.b_reasons, u.unrecognized, None, u.as_of, u.valid_days,
    )  # fmt: skip


# ---- 第 1.3 节：预期单位清单与检查 ----


def expected_units(
    security: str, event_set: str, events: Iterable[SevereEvent]
) -> tuple[UnitId, ...]:
    """只由事件链生成，与任何方法无关：全部事件，以及全部非空的依赖区间。"""
    out: list[UnitId] = []
    for e in events:
        out.append(event_unit_id(security, event_set, e))
        iv = interval_unit_id(security, event_set, e)
        if iv is not None:
            out.append(iv)
    return tuple(out)


def class_a_set(units: Iterable[H1Unit]) -> frozenset[tuple[UnitId, frozenset[Category]]]:
    return frozenset((u.unit_id, u.categories) for u in units if u.result is UnitResult.CLASS_A)


def check_units(
    expected: Sequence[UnitId],
    expected_methods: Sequence[str],
    results: Mapping[str, Sequence[H1Unit]],
) -> frozenset[tuple[UnitId, frozenset[Category]]]:
    """方法集合、覆盖（不漏、不重、不多）、甲类集合的跨方法一致（含 B3）。不一致即抛错。

    返回各方法共同的甲类集合（单位身份 + 原因类别）；预期清单为空时返回空集合。"""
    if len(set(expected_methods)) != len(expected_methods):
        raise H1ContractError("预期方法名单有重复")
    if set(results) != set(expected_methods):
        missing = sorted(set(expected_methods) - set(results))
        extra = sorted(set(results) - set(expected_methods))
        raise H1ContractError(f"方法集合与预期名单不同：缺 {missing}，多 {extra}")
    want = Counter(expected)
    if any(n > 1 for n in want.values()):
        raise H1ContractError("预期单位清单有重复")
    common: frozenset[tuple[UnitId, frozenset[Category]]] | None = None
    first = ""
    for name in expected_methods:
        units = results[name]
        if any(u.method != name for u in units):
            raise H1ContractError(f"{name}: 结果中混有其他方法的单位")
        got = Counter(u.unit_id for u in units)
        dup = [k for k, n in got.items() if n > 1]
        if dup:
            raise H1ContractError(f"{name}: 单位重复 {dup[0]}")
        if set(got) - set(want):
            raise H1ContractError(f"{name}: 有清单外的单位 {sorted(set(got) - set(want))[0]}")
        if set(want) - set(got):
            raise H1ContractError(f"{name}: 漏了单位 {sorted(set(want) - set(got))[0]}")
        a = class_a_set(units)
        if common is None:
            common, first = a, name
        elif a != common:
            raise H1ContractError(f"{name} 的甲类集合与 {first} 不同")
    return common or frozenset()


# ---- 第 2 节：汇总、比例与上限 ----


class DatasetRole(StrEnum):
    EXPLORATION = "EXPLORATION"  # 探索集
    SELECTION = "SELECTION"  # 选参集
    VALIDATION = "VALIDATION"  # 验证集


class CapsNotApproved(StrEnum):
    NOT_APPROVED = "NOT_APPROVED"


CAPS_NOT_APPROVED = CapsNotApproved.NOT_APPROVED


def _cap_ratio(name: str, value: object) -> tuple[int, int]:
    if type(value) is not Decimal:  # 拒绝 float、bool、int：不做自动转换
        raise H1ContractError(f"{name} 必须是 Decimal，实际 {type(value).__name__}")
    assert isinstance(value, Decimal)
    if not value.is_finite():
        raise H1ContractError(f"{name} 必须是有限值")
    if value < 0 or value > 1:
        raise H1ContractError(f"{name} 必须在 [0, 1] 之内")
    return value.as_integer_ratio()


@dataclass(frozen=True, slots=True)
class Caps:
    """已批准的上限（由调用方传入；本模块不设默认值）。"""

    u_class_a: Decimal
    u_delisted: Decimal

    def __post_init__(self) -> None:
        _cap_ratio("U_甲", self.u_class_a)
        _cap_ratio("U_退市", self.u_delisted)


@dataclass(frozen=True, slots=True)
class Ratio:
    numerator: int
    denominator: int

    @property
    def determinable(self) -> bool:
        return self.denominator != 0

    @property
    def q12(self) -> Decimal | None:
        """按 decimal_ratio.q12 量化的 12 位小数；分母为 0 时不可判定（None），不填 0。"""
        if not self.determinable:
            return None
        return ratio(Decimal(self.numerator), Decimal(self.denominator))

    def within(self, cap: tuple[int, int]) -> bool:
        """精确比较：n × b ≤ a × d。调用前须先判分母。"""
        a, b = cap
        return self.numerator * b <= a * self.denominator


class CapConclusion(StrEnum):
    NOT_APPROVED_DISPLAY_ONLY = "上限未批准，只展示"
    WITHIN = "上限内"
    INSUFFICIENT_EVIDENCE = "证据不足"
    UNDETERMINABLE = "不可判定"


@dataclass(frozen=True, slots=True)
class H1Summary:
    checkable_events: int
    checkable_intervals: int
    not_judged: int
    ok: int
    violation: int
    class_b: Mapping[ClassBReason, int]  # 乙类按原因分列（一个单位可有两个原因）
    class_b_units: int
    class_a_events: int
    class_a_intervals: int
    event_categories: Mapping[Category, int]
    interval_categories: Mapping[Category, int]
    violations_for_eligibility: int  # VIOLATION + 乙类
    sensitivity_violations: int  # 违反数 + 甲类单位数（只作描述）
    event_class_a: Ratio  # ①
    mixed_class_a: Ratio  # ②
    event_delisted: Ratio  # ③
    mixed_delisted: Ratio  # ④
    cap_conclusion: CapConclusion
    exceeded: tuple[str, ...]  # 超过上限的比例（① ② ③ ④）

    @property
    def checkable(self) -> int:
        return self.checkable_events + self.checkable_intervals


def summarize(
    units: Sequence[H1Unit],
    kind: MethodKind,
    period: tuple[date, date],
    role: DatasetRole,
    caps: Caps | CapsNotApproved,
) -> H1Summary:
    """一个事件集、一个方法、一个期间 [start, end_exclusive) 的汇总（第 2 节）。

    caps 必填，可以显式传"未批准"；选参集与验证集的角色下传"未批准"即抛错。
    上限判定结论与 H1 违反数分别输出："上限内"不说明 H1 通过。"""
    if kind is not MethodKind.DAILY:
        raise H1ContractError("无逐日状态的方法（B3）不进入汇总")
    if any(u.result is UnitResult.NOT_APPLICABLE_NO_DAILY_STATE for u in units):
        raise H1ContractError("汇总中出现了无逐日状态方法的单位")
    if not isinstance(role, DatasetRole):
        raise H1ContractError("必须给出数据集的角色")
    if isinstance(caps, CapsNotApproved):
        if role is not DatasetRole.EXPLORATION:
            raise H1ContractError("选参集与验证集必须传入已批准的上限，才能计算比例")
        cap_pair = None
    elif isinstance(caps, Caps):
        cap_pair = (_cap_ratio("U_甲", caps.u_class_a), _cap_ratio("U_退市", caps.u_delisted))
    else:
        raise H1ContractError("caps 必须是 Caps 或 CAPS_NOT_APPROVED")
    start, end = period
    inside = [u for u in units if start <= u.as_of < end]
    events = [u for u in inside if u.kind is UnitKind.EVENT]
    intervals = [u for u in inside if u.kind is UnitKind.INTERVAL]
    checkable_e = [u for u in events if u.result is not UnitResult.NOT_JUDGED]
    a_e = [u for u in events if u.result is UnitResult.CLASS_A]
    a_i = [u for u in intervals if u.result is UnitResult.CLASS_A]
    b_units = [u for u in inside if u.result is UnitResult.CLASS_B]
    b_count: Counter[ClassBReason] = Counter()
    for u in b_units:
        b_count.update(u.b_reasons)
    ev_cats: Counter[Category] = Counter()
    for u in a_e:
        ev_cats.update(u.categories)
    iv_cats: Counter[Category] = Counter()
    for u in a_i:
        iv_cats.update(u.categories)
    violation = sum(1 for u in events if u.result is UnitResult.VIOLATION)
    violations = violation + len(b_units)
    n_e, n_all = len(checkable_e), len(checkable_e) + len(intervals)
    delisted_e = sum(1 for u in a_e if Category.DELISTED in u.categories)
    delisted_i = sum(1 for u in a_i if Category.DELISTED in u.categories)
    r1 = Ratio(len(a_e), n_e)
    r2 = Ratio(len(a_e) + len(a_i), n_all)
    r3 = Ratio(delisted_e, n_e)
    r4 = Ratio(delisted_e + delisted_i, n_all)
    exceeded: list[str] = []
    if cap_pair is None:
        conclusion = CapConclusion.NOT_APPROVED_DISPLAY_ONLY
    else:
        u_a, u_d = cap_pair
        checks = (("①", r1, u_a), ("②", r2, u_a), ("③", r3, u_d), ("④", r4, u_d))
        undeterminable = any(not r.determinable for _, r, _ in checks)
        exceeded = [n for n, r, cap in checks if r.determinable and not r.within(cap)]
        if undeterminable:  # 任一比例不可判定：不称上限内（HA04）
            conclusion = CapConclusion.UNDETERMINABLE
        elif exceeded:
            conclusion = CapConclusion.INSUFFICIENT_EVIDENCE
        else:
            conclusion = CapConclusion.WITHIN
    return H1Summary(
        checkable_events=n_e,
        checkable_intervals=len(intervals),
        not_judged=sum(1 for u in events if u.result is UnitResult.NOT_JUDGED),
        ok=sum(1 for u in events if u.result is UnitResult.OK),
        violation=violation,
        class_b=dict(b_count),
        class_b_units=len(b_units),
        class_a_events=len(a_e),
        class_a_intervals=len(a_i),
        event_categories=dict(ev_cats),
        interval_categories=dict(iv_cats),
        violations_for_eligibility=violations,
        sensitivity_violations=violations + len(a_e) + len(a_i),
        event_class_a=r1,
        mixed_class_a=r2,
        event_delisted=r3,
        mixed_delisted=r4,
        cap_conclusion=conclusion,
        exceeded=tuple(exceeded),
    )
