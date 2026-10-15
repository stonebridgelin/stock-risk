"""T08a B-02：确认线缺失抛错；不变量第 2 条（K 日后 box_high 必须有值）与第 6 条（确认消失或
周期重置时当日必须有终止事件）。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from stock_risk.domain.lifecycle.invariants import check_invariants
from stock_risk.domain.lifecycle.model import EventType, Level
from stock_risk.domain.lifecycle.params import AtrBasis, Confirm, ReleasePath
from stock_risk.domain.numeric import InvalidInputError
from tests.unit.lifecycle.harness import Run, params, s_run
from tests.unit.lifecycle.test_fixtures_f import run_path
from tests.unit.lifecycle.test_fixtures_s1_s9 import S1_ROWS
from tests.unit.lifecycle.v08_harness import V0, Day, Prior, Table, run_table
from tests.unit.lifecycle.v08_harness import params as v08_params

E = EventType
ALERT_PRIOR = Prior(Level.ALERT, l_ep=80, entry_path="S", delta_start=20, since_obs=5)


def test_box_missing_line_raises() -> None:
    """BOX：since_obs = 5 > K 而 box_high 为空（前态缺线），不得静默视为确认不成立。"""
    with pytest.raises(InvalidInputError, match="BOX 的确认线缺失"):
        run_table(Table([Day("t", 85)]), v08_params(V0, Confirm.BOX), ALERT_PRIOR)


def test_box_with_line_confirms() -> None:
    run = run_table(
        Table([Day("t", 85)]), v08_params(V0, Confirm.BOX), replace(ALERT_PRIOR, box_high=84)
    )
    assert run.rec("t").confirmed is True and run.rec("t").box_high == 84
    assert [e.type for e in run.events("t")] == [E.RECOVERY_OBS]
    assert run.invariants() == []


def test_box_roll_missing_line_raises() -> None:
    """BOX_ROLL：今日之前不足 K 条有效日线，确认线不可得。"""
    with pytest.raises(InvalidInputError, match="BOX_ROLL 的确认线缺失"):
        run_table(Table([Day("t", 85)]), v08_params(V0, Confirm.BOX_ROLL), ALERT_PRIOR)


def test_box_line_not_needed_before_k() -> None:
    """since_obs ≤ K 时不需要确认线，缺线不报错。"""
    run = run_table(
        Table([Day("t", 85)]), v08_params(V0, Confirm.BOX_ROLL), replace(ALERT_PRIOR, since_obs=1)
    )
    assert run.rec("t").confirmed is False


def _drop(run: Run, label: str, field: str | None = None, kind: EventType | None = None) -> list:
    out = []
    for r in run.result.results:
        if r.record.day == run.labels[label]:
            if field is not None:
                r = replace(r, record=replace(r.record, **{field: None}))
            if kind is not None:
                r = replace(r, events=tuple(e for e in r.events if e.type is not kind))
        out.append(r)
    return out


def test_invariant2_box_high_required_after_k() -> None:
    rows = [("b0", 80, None), ("b1", 81, None), ("b2", 82, None), ("b3", 83, None)]
    rows += [("b4", 84, None), ("b5", 85, None)]
    run = s_run(rows, params(Confirm.BOX))
    assert run.invariants() == []
    bad = check_invariants(_drop(run, "b4", field="box_high"), run.params)
    assert any("不变量2：since_obs ≥ K 时 box_high 为空" in x for x in bad)


@pytest.mark.parametrize("confirm", [Confirm.SMA, Confirm.BOX_ROLL])
def test_b03_invariant2_box_high_empty_unless_box(confirm: Confirm) -> None:
    """Gemini 第二阶段 B-03：confirm ≠ BOX 时 box_high 必须为空。"""
    rows = [("b0", 80, 83), ("b1", 81, 83), ("b2", 82, 83), ("b3", 83, 83), ("b4", 84, 83)]
    run = s_run(rows, params(confirm))
    assert run.invariants() == []
    bad = []
    for r in run.result.results:
        if r.record.day == run.labels["b4"]:
            r = replace(r, record=replace(r.record, box_high=84.0))
        bad.append(r)
    out = check_invariants(bad, run.params)
    assert any(f"不变量2：confirm = {confirm} 时 box_high 有值" in x for x in out)


def test_invariant6_revoke_required_when_confirmation_disappears() -> None:
    """S7：x3 确认，x4 收盘跌破 SMA10 并撤销；去掉撤销事件后应报告违反。"""
    rows = [("x0", 80, None), ("x1", 81, 83), ("x2", 82, 82.5), ("x3", 83, 82)]
    rows += [("x4", 82, 82.5), ("x5", 84, 82.8)]
    run = s_run(rows, params())
    assert run.invariants() == []
    bad = check_invariants(_drop(run, "x4", kind=E.RECOVERY_REVOKED), run.params)
    assert any("不变量6：上一交易日的确认在当日消失或被重置" in x for x in bad)


def test_invariant6_fail_required_on_reset() -> None:
    """F3 BOX：f6 确认，f7 新低重置并以 RECOVERY_FAIL 结束确认；去掉后应报告违反。"""
    run = run_path("F3", Confirm.BOX, ReleasePath.TWO_STAGE, AtrBasis.PEAK)
    assert run.invariants() == []
    bad = check_invariants(_drop(run, "f7", kind=E.RECOVERY_FAIL), run.params)
    assert any("不变量6：上一交易日的确认在当日消失或被重置" in x for x in bad)


def test_invariant6_not_required_on_release_day() -> None:
    """S1：s8 确认成立，s9 解除（当日不发进度事件），不算违反。"""
    run = s_run(S1_ROWS, params())
    assert run.rec("s8").prev_confirm and [e.type for e in run.events("s9")] == [E.RELEASE]
    assert run.invariants() == []
