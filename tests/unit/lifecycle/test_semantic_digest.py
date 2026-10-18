"""T10-M0b 第 1 节：全语义摘要。期望全部来自任务卡的编码规则，不用实现的输出倒填。"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal
from types import MappingProxyType

import pytest

from stock_risk.application.replay_run_types import discrete_digest
from stock_risk.domain.lifecycle.model import Display, Level
from stock_risk.domain.lifecycle.params import AtrBasis, Confirm, ReleasePath
from stock_risk.domain.lifecycle.runner import RunResult
from stock_risk.domain.lifecycle.semantic_digest import CanonicalError, canonical, semantic_digest
from tests.unit.lifecycle.test_fixtures_f import run_path

# ---- 1. 编码单元测试（手写期望字符串） ----


def test_scalar_encodings() -> None:
    assert canonical(None) == "null"
    assert canonical(True) == "true"
    assert canonical(1) == '{"i":"1"}'
    assert canonical(0.1) == '{"f":"0x1.999999999999ap-4"}'  # IEEE 754 双精度 0.1
    assert canonical(-0.0) != canonical(0.0)
    assert canonical(Decimal("1.50")) == '{"d":"1.50"}'
    assert canonical(Decimal("1.50")) != canonical(Decimal("1.5"))
    assert canonical(date(2019, 12, 31)) == '{"date":"2019-12-31"}'
    assert canonical("正常") == '"正常"'


def test_enum_encodings_include_module_and_qualname() -> None:
    assert canonical(Level.WATCH) != canonical(1)
    assert canonical(Display.NORMAL) != canonical("正常")
    for member in (Level.WATCH, Display.NORMAL):
        enc = canonical(member)
        assert "stock_risk.domain.lifecycle.model." in enc and type(member).__qualname__ in enc
        assert enc.endswith(f'.{member.name}"}}')


def test_container_encodings() -> None:
    assert canonical(frozenset({"b", "a"})) == canonical(frozenset({"a", "b"}))
    assert canonical({"x": None}) != canonical({})
    assert canonical((1,)) != canonical(1)
    assert canonical([1]) != canonical((1,))
    assert canonical(frozenset({1})) != canonical({1})
    assert len({canonical(1), canonical(1.0), canonical(True)}) == 3
    assert canonical((1,)) == '{"tuple":[{"i":"1"}]}'
    assert canonical([1]) == '{"list":[{"i":"1"}]}'
    assert canonical({"b": 2, "a": 1}) == '{"map":[["a",{"i":"1"}],["b",{"i":"2"}]]}'


class _Unlisted:
    pass


@pytest.mark.parametrize(
    "value",
    [
        datetime(2019, 1, 2, 3, 4),
        float("nan"),
        float("inf"),
        float("-inf"),
        Decimal("NaN"),
        Decimal("Infinity"),
        object(),
        _Unlisted(),
        MappingProxyType({"a": 1}),
    ],
)
def test_rejected_values_report_field_path(value: object) -> None:
    with pytest.raises(CanonicalError) as e:
        canonical({"outer": [(1, value)]})
    assert "['outer'][0][1]" in str(e.value)


# ---- 2. 敏感性（I-18） ----


def _base() -> RunResult:
    run = run_path("F2", Confirm.SMA, ReleasePath.DIRECT, AtrBasis.MIN)
    return run.result


def _replace_step(result: RunResult, i: int, **record_changes: object) -> RunResult:
    step = result.results[i]
    new = replace(step, record=replace(step.record, **record_changes))
    return replace(result, results=(*result.results[:i], new, *result.results[i + 1 :]))


def _mutations(result: RunResult) -> dict[str, RunResult]:
    idx_rc = next(i for i, r in enumerate(result.results) if r.record.rc is not None)
    idx_atr = next(i for i, r in enumerate(result.results) if r.record.atr_ep is not None)
    idx_ev = next(i for i, r in enumerate(result.results) if any(e.basis for e in r.events))
    step = result.results[idx_ev]
    ev = next(e for e in step.events if e.basis)
    k = next(iter(ev.basis))
    new_basis = {**ev.basis, k: "改动"}
    events = tuple(replace(e, basis=new_basis) if e is ev else e for e in step.events)
    with_basis = replace(
        result,
        results=(
            *result.results[:idx_ev],
            replace(step, events=events),
            *result.results[idx_ev + 1 :],
        ),
    )
    atr = result.results[idx_atr].record.atr_ep
    assert atr is not None
    return {
        "rc": _replace_step(result, idx_rc, rc=result.results[idx_rc].record.rc + 1),
        "atr_ep": _replace_step(result, idx_atr, atr_ep=math.nextafter(atr, math.inf)),
        "basis": with_basis,
        "final_state": replace(
            result, final_state=replace(result.final_state, m=result.final_state.m + 1)
        ),
        "censored_episode": replace(result, censored_episode=(result.censored_episode or "") + "X"),
    }


def test_semantic_digest_detects_every_field_but_discrete_does_not() -> None:
    base = _base()
    d0, disc0 = semantic_digest(base), discrete_digest(base)
    assert len(d0) == 64 and d0 == d0.lower()
    for name, changed in _mutations(base).items():
        assert semantic_digest(changed) != d0, name
        assert discrete_digest(changed) == disc0, name  # 旧摘要不含这些字段：确有缺口
    assert semantic_digest(base) == d0  # 原对象没有被改动


def test_digest_independent_of_set_construction_order() -> None:
    base = _base()
    i = next(i for i, r in enumerate(base.results) if len(r.record.evidence) >= 1)
    ev = base.results[i].record.evidence
    rebuilt = frozenset(reversed(sorted(ev)))
    same = _replace_step(base, i, evidence=rebuilt)
    assert semantic_digest(same) == semantic_digest(base)
    a = {"k": frozenset({"x", "y", "z"}), "j": {3, 1, 2}}
    b = {"j": {2, 3, 1}, "k": frozenset({"z", "y", "x"})}
    assert canonical(a) == canonical(b)


def test_real_fixture_output_is_canonicalizable() -> None:
    """生命周期夹具的完整输出没有被拒绝的值（时间戳、非有限数、未列出的类型）。"""
    for path in ("F1", "F2", "F3", "F4", "F5"):
        run = run_path(path, Confirm.BOX, ReleasePath.TWO_STAGE, AtrBasis.PEAK)
        assert len(semantic_digest(run.result)) == 64
