"""17 §2.2 / D-08：单个事件的逐有效日状态转移（内部序号不对外输出）。"""

from dataclasses import dataclass, replace
from decimal import Decimal

from stock_risk.domain.evaluation.decimal_ratio import drawdown, ratio
from stock_risk.domain.evaluation.event_model import EventKind, EventParameters, EventSetSpec
from stock_risk.domain.evaluation.params import CONFIRM_DAYS, EVENT_CAP, QUIET_DAYS


@dataclass(frozen=True, slots=True)
class Progress:
    s: int
    p: int
    low: Decimal
    anchor: int
    a_s: Decimal | None
    cap_continuation: bool
    q: int = 0
    e: int | None = None
    reason: str | None = None
    confirm: int | None = None
    flags: frozenset[str] = frozenset()
    merges: int = 0


def distance(close: Decimal, peak: Decimal, spec: EventSetSpec, atr: Decimal | None) -> Decimal:
    if spec.kind is EventKind.PCT:
        return drawdown(close, peak)
    assert atr is not None and atr > 0, "E_atr 所需的 ATR20 不可计算"
    return ratio(peak - close, atr)


def advance(
    state: Progress,
    j: int,
    closes: tuple[Decimal, ...],
    spec: EventSetSpec,
    params: EventParameters,
) -> Progress:
    """开始日不调用；顺序为更新低点，再执行待确认或进行中的终止性分支。"""
    current = state
    if closes[j] <= state.low:  # 相同低点移动恢复锚，报告谷底另取最早日（D-12）
        current = replace(state, low=closes[j], anchor=j)
    n = j - state.s + 1
    if state.e is not None:
        return _confirming(current, state.low, j, closes, n)
    if closes[j] >= closes[state.p]:
        return replace(current, e=j, reason="a", confirm=j)
    if n == EVENT_CAP:
        return replace(current, e=j, reason="c", confirm=j, flags=frozenset({"cap"}))
    recovery = ratio(closes[j] - current.low, closes[state.p] - current.low)
    b = distance(closes[j], closes[state.p], spec, state.a_s)
    m = max(closes[max(current.anchor, j - params.w_q + 1) : j + 1])
    v = distance(closes[j], m, spec, state.a_s)
    quiet = (recovery >= params.rho_end or b < spec.threshold / 2) and v < spec.threshold / 2
    q = state.q + 1 if quiet else 0
    if q == QUIET_DAYS:
        return replace(current, q=q, e=j, reason="b")
    return replace(current, q=q)


def _confirming(
    state: Progress, tentative_low: Decimal, j: int, closes: tuple[Decimal, ...], n: int
) -> Progress:
    assert state.e is not None
    if closes[j] < tentative_low:  # 并入优先于确认与上限，q 当日清零
        merged = replace(state, e=None, reason=None, q=0, merges=state.merges + 1)
        if n == EVENT_CAP:
            return replace(merged, e=j, reason="c", confirm=j, flags=frozenset({"cap"}))
        return merged
    if closes[j] >= closes[state.p]:
        return replace(state, confirm=j, flags=frozenset({"peak_in_confirm"}))
    if n == EVENT_CAP:
        return replace(state, confirm=j, flags=frozenset({"confirm_cut_by_cap"}))
    if j - state.e == CONFIRM_DAYS:
        return replace(state, confirm=j)
    return state
