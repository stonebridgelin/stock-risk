"""T09-E2 开工补充第 1 节第 3 条的属性测试。期望来自规格推导，不来自实现输出。

- I-11：每个事件在每个方法、每个口径下恰好落入一类（C、N、U，或 EXCLUDED），各类互斥且覆盖全部事件；
  EXCLUDED 必有原因代码、指标为空，纳入的类别必有 S1、S1w。
- I-14：同一窗口下 S2⁺ ≥ S2（h⁺ ≥ h 同理）。17 §3B：S2⁺ = (a + u + x) / (e + u + x)，S2 = a / e，
  a ≤ e，分子分母加同一个非负数，比例不减。
- I-15：解除集合固定时，按补足口径（不完整窗口计为不利，17 §3B）：
  ① 同一次解除，补足后的不利对窗口长度单调（Q ≤ Q_L ⇒ 不利⁺(Q) ⇒ 不利⁺(Q_L)，窗口逐日嵌套）；
  ② 补足后的不利率 p⁺ ≥ 主口径 p（加入的项都计为 1）。
  建议稿第 5.7 节的原文不在仓库中，以上按 17 号推导，出入写入回执。
方法的逐日状态用固定种子的伪随机区段生成（与实现无关的输入）。
"""

import random
from decimal import Decimal

import pytest

from stock_risk.domain.evaluation.method_series import AlertLevel, MethodEvent, MethodSeries
from stock_risk.domain.evaluation.outcomes import Caliber, OutcomeClass, classify
from stock_risk.domain.evaluation.release_windows import AdverseThresholds, post_signal_window
from stock_risk.domain.evaluation.series import EvalSeries
from stock_risk.domain.evaluation.windows import (
    RecoveryStatus,
    h_window,
    recovery_ref,
    s2_window,
)
from tests.unit.evaluation.e2_helpers import WINDOW
from tests.unit.evaluation.helpers import load_fixtures, make_series, run_case

CASES = [
    ("A", k) for k in ("A01", "A02", "A07", "A29b", "A30", "A31", "A39b", "A_DB", "A_M3", "A_P10")
] + [("B", k) for k in ("B13", "B15", "B20a", "B20c", "B21", "B21m", "B33", "B35", "B36", "B40")]


def random_method(series: EvalSeries, seed: int) -> MethodSeries:
    rng = random.Random(seed)
    days = series.calendar.days
    levels: list[AlertLevel | None] = [None] * len(days)
    episodes: list[str | None] = [None] * len(days)
    events: list[MethodEvent] = []
    i, n = 0, 0
    while i < len(days):
        if rng.random() < 0.08:
            n += 1
            length = rng.randint(1, 40)
            level = rng.choice(list(AlertLevel))
            events.append(MethodEvent("START", days[i], f"e{n}"))
            for j in range(i, min(i + length, len(days))):
                if rng.random() < 0.05:
                    level = AlertLevel.ALERT
                levels[j], episodes[j] = level, f"e{n}"
            i += length
            if i < len(days):
                events.append(MethodEvent("RELEASE", days[i], f"e{n}"))
        i += 1
    return MethodSeries(f"R{seed}", days, tuple(levels), tuple(episodes), tuple(events))


def _case(group: str, case_id: str):  # type: ignore[no-untyped-def]
    case = load_fixtures()[group][case_id]
    return run_case({"event_set": {"kind": "pct", "T": "0.2"}, **case})


@pytest.mark.parametrize(("group", "case_id"), CASES)
def test_I11_outcome_classes_exclusive_and_exhaustive(group: str, case_id: str) -> None:
    series, chain = _case(group, case_id)
    for seed in range(5):
        method = random_method(series, seed)
        for caliber in Caliber:
            counts = dict.fromkeys(OutcomeClass, 0)
            for event in chain.events:
                o = classify(series, method, event, caliber)
                counts[o.outcome_class] += 1
                if o.outcome_class is OutcomeClass.EXCLUDED:
                    assert o.reasons and o.s1 is None and o.s1w is None and o.trough is None
                else:
                    assert not o.reasons and o.s1 is not None and o.s1w is not None
                    assert (o.outcome_class is OutcomeClass.U) == (o.f0 is None)
                    assert Decimal(0) <= o.s1 <= Decimal(1)
            assert sum(counts.values()) == len(chain.events)
            main = sum(e.is_main for e in chain.events)
            included = len(chain.events) - counts[OutcomeClass.EXCLUDED]
            if caliber is Caliber.MAIN:
                assert included == main
            else:
                assert included == main + sum(e.is_prefix_complete for e in chain.events)


@pytest.mark.parametrize(("group", "case_id"), CASES)
def test_I14_supplemented_not_below_main(group: str, case_id: str) -> None:
    series, chain = _case(group, case_id)
    for seed in range(5):
        method = random_method(series, seed)
        for event in (e for e in chain.events if e.is_main):
            rec = recovery_ref(series, chain, event, WINDOW)
            if rec.status is RecoveryStatus.FOUND:
                w = s2_window(series, method, chain, event, rec, WINDOW).window
            else:
                w = h_window(series, method, chain, event, rec, WINDOW)
            for value in (w.value, w.value_plus):
                assert value is None or Decimal(0) <= value <= Decimal(1)
            if w.value is not None and w.value_plus is not None:
                assert w.value_plus >= w.value, (case_id, seed, event.p)


def _adverse_plus(w) -> bool:  # type: ignore[no-untyped-def]
    """补足口径：不完整窗口计为不利（17 §3B）。"""
    return (not w.complete) or bool(w.adverse)


@pytest.mark.parametrize("seed", range(4))
def test_I15_supplemented_adverse_monotone(seed: int) -> None:
    rng = random.Random(100 + seed)
    case = load_fixtures()["B"]["B41"]["B41e"]  # 含 DATA_BREAK 的序列，窗口可能不完整
    series = make_series(case)
    thresholds = AdverseThresholds(Decimal("0.03"), Decimal("2"))
    valid = [d for d, b in zip(series.calendar.days, series.bars, strict=True) if b is not None]
    releases = rng.sample([d for d in valid if series.atr20[series.index(d)] is not None], 15)
    main_hits = main_n = plus_hits = 0
    for r in releases:
        l_ep = Decimal(str(rng.choice([95, 99.6, 100])))
        q = post_signal_window(series, r, 20, l_ep, thresholds)
        q_l = post_signal_window(series, r, 60, l_ep, thresholds)
        assert not _adverse_plus(q) or _adverse_plus(q_l)  # ①
        if q.complete:
            main_n += 1
            main_hits += bool(q.adverse)
        plus_hits += _adverse_plus(q)
    assert 0 < main_n < len(releases)  # 样本中既有完整窗口，也有跨断点的不完整窗口
    if main_n:  # ② p⁺ = plus_hits / 15 ≥ p = main_hits / main_n
        assert Decimal(plus_hits) / Decimal(len(releases)) >= Decimal(main_hits) / Decimal(main_n)
