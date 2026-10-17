"""17 §3D、§3.1、§5.3、§7、§10A：汇总、最小样本、门槛判定、集中度、两段检查与 X 校准。纯函数。

- 门槛值（m、p、p_L、s、o_max、o_tail、h_tail、F）与最小样本只由调用方传入，本模块不写默认值；
- 门槛一律为"指标 ≤ 门槛"，按 D-01 的十进制精确比较，不加容差；
- 由样本决定的分母不足 → STOP_INSUFFICIENT；结构上不可能 → NOT_APPLICABLE；
  由候选决定的分母不足 → CANDIDATE_INSUFFICIENT；验证期出现以上情形 → UNDETERMINED（17 §10A）；
- h_tail 的分组例外（17 §3.1）：三个组各自都有至少规定数目的主口径合格窗口时按组过关，
  否则只按全体判定并报告各组窗口数；主口径与补足口径共用同一判定。
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, Decimal, localcontext
from enum import StrEnum

from stock_risk.domain.evaluation.decimal_ratio import EVALUATION_CONTEXT, q12, ratio
from stock_risk.domain.evaluation.event_model import SevereEvent
from stock_risk.domain.evaluation.membership import Segment
from stock_risk.domain.evaluation.outcomes import OutcomeClass
from stock_risk.domain.evaluation.vol_groups import VolGroup, median, quantile_type7
from stock_risk.domain.numeric import InvalidInputError

ALL = "ALL"  # 全体
GROUPS: tuple[str, ...] = (ALL, VolGroup.LOW, VolGroup.MID, VolGroup.HIGH)
TAIL_P = Decimal("0.9")  # 第 90 百分位（o_tail、h_tail；17 §3.1）
CONCENTRATION_SHARE = Decimal("0.1")  # 贡献 C 类事件最多的 10% 证券（17 §3.1）
GATES = ("m", "p", "p_L", "s", "o_max", "o_tail", "h_tail", "F")


class Period(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    VALIDATION = "VALIDATION"


class SampleStatus(StrEnum):
    OK = "OK"
    STOP_INSUFFICIENT = "STOP_INSUFFICIENT"  # 由样本决定的分母不足：停止评价
    NOT_APPLICABLE = "NOT_APPLICABLE"  # 结构上不可能有样本：只作描述
    CANDIDATE_INSUFFICIENT = "CANDIDATE_INSUFFICIENT"  # 由候选决定的分母不足：候选不合格
    UNDETERMINED = "UNDETERMINED"  # 验证期出现以上情形：无法判定


@dataclass(frozen=True, slots=True)
class MinimumSamples:
    """17 §10A 的最小样本（拟定，待批准）；由调用方传入，没有默认值。"""

    events_overall: int
    events_group: int
    s2_windows: int
    s2_windows_group: int
    s_windows: int
    s_windows_group: int
    h_windows: int  # 整体，以及分组过关时的每组
    releases: int
    releases_group: int
    group_securities: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise InvalidInputError(f"最小样本 {name} 必须为正整数")


@dataclass(frozen=True, slots=True)
class Thresholds:
    """17 §3D 的门槛（待登记）；由调用方传入，没有默认值。"""

    m: Decimal
    p: Decimal
    p_L: Decimal  # noqa: N815 —— 与 17 号记号一致
    s: Decimal
    o_max: Decimal
    o_tail: Decimal
    h_tail: Decimal
    F: Decimal  # noqa: N815

    def __post_init__(self) -> None:
        for name in GATES:
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise InvalidInputError(f"门槛 {name} 必须为有限的 Decimal")


# ---- 最小样本 ----


def sample_status(
    count: int,
    minimum: int,
    period: Period,
    *,
    candidate_denominator: bool = False,
    applicable: bool = True,
) -> SampleStatus:
    """17 §10A。candidate_denominator：分母由候选决定（例如 RELEASE 的完整窗口数）；
    applicable：结构上可能有样本（例如探索集中存在该波动组）。"""
    if not applicable:
        status = SampleStatus.NOT_APPLICABLE
    elif count < minimum:
        status = (
            SampleStatus.CANDIDATE_INSUFFICIENT
            if candidate_denominator
            else SampleStatus.STOP_INSUFFICIENT
        )
    else:
        status = SampleStatus.OK
    if period is Period.VALIDATION and status is not SampleStatus.OK:
        return SampleStatus.UNDETERMINED
    return status


@dataclass(frozen=True, slots=True)
class HTailGrouping:
    grouped: bool  # 按组过关；否则只按全体判定
    overall: SampleStatus
    group_counts: Mapping[str, int]  # 各组主口径窗口数（报告用）


def h_tail_grouping(
    main_counts: Mapping[str, int], overall: int, minimums: MinimumSamples, period: Period
) -> HTailGrouping:
    """17 §3.1 的例外规则：三个组各自都有至少 h_windows 个主口径合格窗口时按组过关。
    每个事件集、每个期间分别调用；主口径与补足口径共用这一结果。"""
    groups = {g: main_counts.get(g, 0) for g in GROUPS[1:]}
    grouped = all(n >= minimums.h_windows for n in groups.values())
    return HTailGrouping(grouped, sample_status(overall, minimums.h_windows, period), groups)


# ---- 指标 ----


def share(hits: int, total: int) -> Decimal | None:
    """占比，量化到 12 位；分母为 0 时为空（不写成 0）。"""
    return ratio(Decimal(hits), Decimal(total)) if total else None


def unalerted_rate(classes: Iterable[OutcomeClass]) -> Decimal | None:
    """m：U 类占比；分母为纳入的事件（EXCLUDED 不计）。"""
    included = [c for c in classes if c is not OutcomeClass.EXCLUDED]
    return share(sum(c is OutcomeClass.U for c in included), len(included))


def s_share(s2_values: Iterable[Decimal]) -> Decimal | None:
    """s：S2 = 1 的窗口占比（调用方按口径给出纳入的窗口）。"""
    values = list(s2_values)
    return share(sum(v == 1 for v in values), len(values))


def adverse_rate(adverse: Iterable[bool]) -> Decimal | None:
    """p、p_L：不利的 RELEASE 占比（补足口径由调用方把不完整窗口记为不利）。"""
    flags = list(adverse)
    return share(sum(flags), len(flags))


def tail(values: Sequence[Decimal]) -> Decimal | None:
    """跨证券或跨窗口的第 90 百分位（第 7 型）；空为 None。"""
    return quantile_type7(values, TAIL_P) if values else None


def middle(values: Sequence[Decimal]) -> Decimal | None:
    """跨证券中位数；空为 None。"""
    return median(values) if values else None


@dataclass(frozen=True, slots=True)
class Concentration:
    top_n: int
    share: Decimal | None  # C 类事件为 0 时为空
    top: tuple[str, ...]


def concentration(c_events: Mapping[str, int]) -> Concentration:
    """17 §3.1：贡献 C 类事件最多的 ⌈10% × 证券数⌉ 只证券所占的比例。并列按证券标识。"""
    n = len(c_events)
    with localcontext(EVALUATION_CONTEXT):
        top_n = int((CONCENTRATION_SHARE * n).to_integral_value(rounding=ROUND_CEILING))
    ranked = sorted(c_events, key=lambda s: (-c_events[s], s))[:top_n]
    total = sum(c_events.values())
    return Concentration(top_n, share(sum(c_events[s] for s in ranked), total), tuple(ranked))


# ---- 门槛判定 ----


@dataclass(frozen=True, slots=True)
class GateCell:
    """一个门槛 × 事件集 × 口径 × 组的输入。事件集、口径不适用时为 None（例如占用率）。"""

    gate: str
    group: str
    value: Decimal | None
    status: SampleStatus
    event_set: str | None = None
    caliber: str | None = None

    def __post_init__(self) -> None:
        if self.gate not in GATES or self.group not in GROUPS:
            raise InvalidInputError(f"门槛或组不合法：{self.gate}、{self.group}")


@dataclass(frozen=True, slots=True)
class GateResult:
    cell: GateCell
    threshold: Decimal
    passed: bool | None  # 样本状态不是 OK，或指标为空时为 None（不判定，不当作通过）
    reason: str | None = None


def judge(cell: GateCell, thresholds: Thresholds) -> GateResult:
    threshold = getattr(thresholds, cell.gate)
    if cell.status is not SampleStatus.OK:
        return GateResult(cell, threshold, None, cell.status.value)
    if cell.value is None:
        return GateResult(cell, threshold, None, "NO_VALUE")
    return GateResult(cell, threshold, cell.value <= threshold)


@dataclass(frozen=True, slots=True)
class GateSummary:
    results: tuple[GateResult, ...]
    all_pass: bool  # 全部判定且全部通过
    evidence_ok: bool  # 没有任何证据不足或无法判定（17 §5.2 第 4 项）
    failed: tuple[GateResult, ...] = field(default=())
    insufficient: tuple[GateResult, ...] = field(default=())


INSUFFICIENT = (
    SampleStatus.STOP_INSUFFICIENT,
    SampleStatus.CANDIDATE_INSUFFICIENT,
    SampleStatus.UNDETERMINED,
)


def judge_all(cells: Iterable[GateCell], thresholds: Thresholds) -> GateSummary:
    """逐格判定。NOT_APPLICABLE 的格只作描述，不参与是否通过。"""
    results = tuple(judge(c, thresholds) for c in cells)
    applicable = [r for r in results if r.cell.status is not SampleStatus.NOT_APPLICABLE]
    failed = tuple(r for r in applicable if r.passed is False)
    insufficient = tuple(r for r in applicable if r.cell.status in INSUFFICIENT)
    undecided = [r for r in applicable if r.passed is None]
    return GateSummary(
        results, not failed and not undecided, not insufficient, failed, insufficient
    )


# ---- X 校准（17 §7） ----


class CalibrationStatus(StrEnum):
    OK = "OK"
    STOP_INSUFFICIENT = "STOP_INSUFFICIENT"


@dataclass(frozen=True, slots=True)
class XCalibration:
    status: CalibrationStatus
    x: Decimal | None
    ratios: Mapping[Decimal, Decimal]  # 参与比较的 X → E_atr 与 E_pct 的事件数之比


def calibration_count(events: Iterable[SevereEvent]) -> int:
    """计数对象：已确认的事件，含上限事件，不含截尾事件（调用方先按开发期、探索集筛选）。"""
    return sum(1 for e in events if e.confirm is not None and not e.censor)


def calibrate_x(pct_events: int, atr_events: Mapping[Decimal, int]) -> XCalibration:
    """取 E_atr 与 E_pct 事件数之比最接近 1 的 X；并列取较小的 X；事件数为 0 的 X 不参与。
    E_pct 为 0 或全部 X 下 E_atr 都为 0 时停止，报告证据不足。"""
    used = {x: n for x, n in atr_events.items() if n > 0}
    if pct_events <= 0 or not used:
        return XCalibration(CalibrationStatus.STOP_INSUFFICIENT, None, {})
    with localcontext(EVALUATION_CONTEXT):
        ratios = {x: Decimal(n) / Decimal(pct_events) for x, n in sorted(used.items())}
        best = min(ratios, key=lambda x: (abs(ratios[x] - 1), x))
    return XCalibration(CalibrationStatus.OK, best, {x: q12(r) for x, r in ratios.items()})


# ---- 开发期两段检查（17 §5.3） ----


@dataclass(frozen=True, slots=True)
class SegmentSplit[V]:
    front: tuple[V, ...]
    back: tuple[V, ...]
    neither: int  # 两段都不使用的个数（按指标报告）


def split_segments[V](items: Iterable[tuple[Segment, V]]) -> SegmentSplit[V]:
    """按 membership.attribute 的归段结果分开；NEITHER 只计数。"""
    pairs = list(items)
    return SegmentSplit(
        tuple(v for s, v in pairs if s is Segment.FRONT),
        tuple(v for s, v in pairs if s is Segment.BACK),
        sum(1 for s, _ in pairs if s is Segment.NEITHER),
    )


__all__ = [
    "ALL",
    "GATES",
    "GROUPS",
    "CalibrationStatus",
    "Concentration",
    "GateCell",
    "GateResult",
    "GateSummary",
    "HTailGrouping",
    "MinimumSamples",
    "Period",
    "SampleStatus",
    "SegmentSplit",
    "Thresholds",
    "XCalibration",
    "adverse_rate",
    "calibrate_x",
    "calibration_count",
    "concentration",
    "h_tail_grouping",
    "judge",
    "judge_all",
    "middle",
    "s_share",
    "sample_status",
    "share",
    "split_segments",
    "tail",
    "unalerted_rate",
]
