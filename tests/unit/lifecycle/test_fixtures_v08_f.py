"""13 号第 6 节 F2x、F3x、F6、F7（完整行情，按第 2 节构造，价格十进制精确构造）。

PEAK 与 MIN 都运行。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest

from stock_risk.domain.lifecycle.model import EventType, Level
from stock_risk.domain.lifecycle.params import (
    AtrBasis,
    Confirm,
    LongPath,
    RecScale,
    ReleaseHyst,
    ReleasePath,
    V08Switches,
)
from tests.unit.lifecycle.fields import check_days, released, reset_at, started
from tests.unit.lifecycle.test_fixtures_f import expected_days
from tests.unit.lifecycle.v08_harness import V0, B, f_run_decimal, params

E = EventType
BASES = [AtrBasis.PEAK, AtrBasis.MIN]


def dec(xs: list[float | str]) -> list[Decimal]:
    return [Decimal(str(x)) for x in xs]


def event_days(run) -> dict[str, list[EventType]]:  # type: ignore[no-untyped-def]
    inv = {d: k for k, d in run.labels.items()}
    return {
        inv[r.record.day]: sorted(e.type for e in r.events) for r in run.result.results if r.events
    }


F2_PATH = [
    98,
    96,
    94,
    92,
    90,
    88,
    86,
    84,
    82,
    80,
    81,
    88,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
]
F2X_SW = replace(B, rec_scale=RecScale.PRICE)

# B-01 推导：e4 关注启动；e5–e7 新低（周期 2–4）；e8 新低并 T9 加重（周期 5）；
# e9、e10 新低（周期 6、7）；e12 证据 a（dc 1），e13 中断；e15 确认；e16 起强恢复（xc）；
# e17 降级（H_d 86，xc 保留 2）；e18 解除
F2X_DAYS = {
    "e4": started("e4", "W"),
    "e5": reset_at("e5", 2),
    "e6": reset_at("e6", 3),
    "e7": reset_at("e7", 4),
    "e8": {**reset_at("e8", 5), "level": "A", "events": [("ESC",)]},
    "e9": reset_at("e9", 6),
    "e10": reset_at("e10", 7),
    "e12": {"dc": 1},
    "e13": {"dc": 0, "events": [("STAB", 7)]},
    "e15": {"prev_confirm": True, "obs_issued": True, "dc": 1, "events": [("OBS", 7, 1)]},
    "e16": {"dc": 2, "xc": 1},
    "e17": {
        "level": "W",
        "downgraded": True,
        "h_d": 86.0,
        "dc": 0,
        "rc": 0,
        "xc": 2,
        "events": [("DOWN",)],
    },
    "e18": released("e18"),
}


@pytest.mark.parametrize("basis", BASES)
def test_F2x_full_expectations(basis: AtrBasis) -> None:
    run = f_run_decimal("e", dec(F2_PATH), params(F2X_SW, Confirm.SMA, ReleasePath.DIRECT, basis))
    assert event_days(run) == {
        "e4": [E.START],
        "e8": [E.ESCALATE],
        "e13": [E.STABILIZED],
        "e15": [E.RECOVERY_OBS],
        "e17": [E.DOWNGRADE],
        "e18": [E.RELEASE],
    }
    assert run.rec("e2").dd_t == pytest.approx(0.04) and run.rec("e3").dd_t == pytest.approx(0.06)
    start = run.event("e4", E.START)
    assert start.basis["reasons"] == ["R1_GE_K1"] and run.rec("e4").level is Level.WATCH
    e4 = run.rec("e4")
    assert (e4.p_ep, e4.atr_ep, e4.entry_path, e4.delta_start) == (100, 2, "S", 8)
    for x, dd in (("e5", 0.10), ("e6", 0.12), ("e7", 0.14)):
        assert run.rec(x).new_low and run.rec(x).level is Level.WATCH
        assert run.rec(x).dd_ep == pytest.approx(dd)
    esc = run.event("e8", E.ESCALATE)
    assert (esc.transition, esc.basis["reason"]) == ("T9", "D_EP_GE_K2")
    assert run.rec("e8").d_ep == pytest.approx(8.0) and run.rec("e8").dd_ep == pytest.approx(0.16)
    assert run.rec("e8").reset_reasons == ("NEW_LOW", "ESCALATE")
    assert run.rec("e10").l_ep == 80
    exp = {  # 证据, dc / xc（今日候选）
        "e11": (set(), 0, 0),
        "e12": ({"a"}, 1, 0),
        "e13": (set(), 0, 0),
        "e14": (set(), 0, 0),
        "e15": ({"b"}, 1, 0),
        "e16": ({"b"}, 2, 1),
        "e17": ({"a", "b"}, 3, 2),
    }
    for x, (ev, dc, xc) in exp.items():
        rec = run.rec(x)
        assert (rec.evidence, rec.dc_cand, rec.xc_cand) == (frozenset(ev), dc, xc), x
    assert run.rec("e15").strong is False and run.rec("e16").strong is True
    assert run.rec("e15").strong_threshold == pytest.approx(84.8)
    assert run.rec("e15").atr_rec == pytest.approx(1.6)
    assert run.event("e17", E.DOWNGRADE).basis["reasons"] == ["a", "b"]
    e17 = run.rec("e17")
    assert (e17.level, e17.downgraded, e17.h_d, e17.dc, e17.rc, e17.xc) == (
        Level.WATCH,
        True,
        86,
        0,
        0,
        2,
    )
    rel = run.event("e18", E.RELEASE)
    assert rel.basis["reasons"] == ["DIRECT_STRONG_RECOVERY"] and rel.transition == "T12"
    e18 = run.rec("e18")
    assert (e18.evidence, e18.rc_cand, e18.xc_cand) == (frozenset({"b′"}), 1, 3)
    assert e18.last_release == run.labels["e18"] and run.rec("e19").theta is None
    check_days(run, F2X_DAYS)
    assert run.invariants() == []


@pytest.mark.parametrize("basis", BASES)
def test_F2x_v0_contrast(basis: AtrBasis) -> None:
    run = f_run_decimal("e", dec(F2_PATH), params(V0, Confirm.SMA, ReleasePath.DIRECT, basis))
    days = event_days(run)
    assert days["e2"] == [E.START] and days["e4"] == [E.ESCALATE]
    assert days["e15"] == [E.RECOVERY_OBS] and days["e17"] == [E.DOWNGRADE]
    assert days["e19"] == [E.RELEASE]
    assert run.event("e17", E.DOWNGRADE).basis["reasons"] == ["b"]
    check_days(run, expected_days("F2", Confirm.SMA, ReleasePath.DIRECT))  # B-01：同 F2 SMA 直接


F3X_PATH = [80, 78, 79, 80, 79.5, 81, 77, 78, 79, 80, 82, 84, 86, 88, 90, 92, 94, 96]

# B-01 推导（取自 13 号 F3x 表；H_d 与确认状态按规则）：f7 新低重置并发出周期 2 的恢复失败；
# f10 重新冻结 box_high = 80；f13 降级（H_d 86），f14、f15 H_d 随收盘上移；f16 解除。
# V0 对照的逐日字段与 B 组相同（只有降级原因不同）。
F3X_DAYS = {
    "f1": started("f1", "A"),
    "f2": reset_at("f2", 2),
    "f5": {"box_high": 80.0, "events": [("STAB", 2)]},
    "f6": {"prev_confirm": True, "obs_issued": True, "dc": 1, "events": [("OBS", 2, 1)]},
    "f7": {**reset_at("f7", 3), "events": [("FAIL", 2)]},
    "f10": {"box_high": 80.0, "events": [("STAB", 3)]},
    "f11": {"prev_confirm": True, "obs_issued": True, "dc": 1, "events": [("OBS", 3, 1)]},
    "f12": {"dc": 2},
    "f13": {"level": "W", "downgraded": True, "h_d": 86.0, "dc": 0, "rc": 0, "events": [("DOWN",)]},
    "f14": {"h_d": 88.0, "rc": 1},
    "f15": {"h_d": 90.0, "rc": 2},
    "f16": released("f16"),
}


@pytest.mark.parametrize("basis", BASES)
def test_F3x_extended_to_release(basis: AtrBasis) -> None:
    run = f_run_decimal("f", dec(F3X_PATH), params(B, Confirm.BOX, ReleasePath.TWO_STAGE, basis))
    eid = run.rec("f1").episode_id
    assert event_days(run) == {
        "f1": [E.START],
        "f5": [E.STABILIZED],
        "f6": [E.RECOVERY_OBS],
        "f7": [E.RECOVERY_FAIL],
        "f10": [E.STABILIZED],
        "f11": [E.RECOVERY_OBS],
        "f13": [E.DOWNGRADE],
        "f16": [E.RELEASE],
    }
    start = run.event("f1", E.START)
    assert start.basis["reasons"] == ["R1_GE_K2"] and run.rec("f1").level is Level.ALERT
    assert run.rec("f1").d_ep == pytest.approx(10) and run.rec("f1").dd_ep == pytest.approx(0.2)
    assert [run.rec(x).cycle_no for x in ("f1", "f2", "f7")] == [1, 2, 3]
    assert run.rec("f5").box_high == 80 and run.rec("f7").box_high is None
    assert run.event("f6", E.RECOVERY_OBS).identity == (eid, 2, 1)
    assert run.event("f7", E.RECOVERY_FAIL).identity == (eid, 2)
    assert run.rec("f10").box_high == 80 and run.event("f10", E.STABILIZED).identity == (eid, 3)
    assert run.event("f11", E.RECOVERY_OBS).identity == (eid, 3, 1)
    assert [run.rec(x).dc_cand for x in ("f11", "f12", "f13")] == [1, 2, 3]
    assert run.rec("f13").evidence == frozenset({"a", "b"})
    assert run.event("f13", E.DOWNGRADE).basis["reasons"] == ["a", "b"]
    assert run.rec("f13").h_d == 86 and run.rec("f13").downgraded
    assert run.rec("f14").strong_threshold == 83 and run.rec("f14").theta == 8
    for x, rc in (("f14", 1), ("f15", 2)):
        assert (run.rec(x).evidence, run.rec(x).rc) == (frozenset({"b′"}), rc)
    assert run.event("f16", E.RELEASE).basis["reasons"] == ["b′"]
    assert run.rec("f16").last_release == run.labels["f16"]
    check_days(run, F3X_DAYS)
    assert run.invariants() == []


def test_F3x_v0_contrast() -> None:
    run = f_run_decimal("f", dec(F3X_PATH), params(V0, Confirm.BOX, ReleasePath.TWO_STAGE))
    assert run.event("f13", E.DOWNGRADE).basis["reasons"] == ["b"]
    assert run.event("f16", E.RELEASE).basis["reasons"] == ["b′"]
    check_days(run, F3X_DAYS)


def decline_days(t_start: int, level: str, t_alert: int | None, last: int) -> dict:
    """B-01 推导（F6、F7）：启动后收盘逐日创新低，每日观察重置，周期依次加一；只在 t_alert 加重。"""
    days = {f"t{t_start}": started(f"t{t_start}", level)}
    for t in range(t_start + 1, last + 1):
        days[f"t{t}"] = reset_at(f"t{t}", t - t_start + 1)
    if t_alert is not None:
        days[f"t{t_alert}"] |= {"level": "A", "events": [("ESC",)]}
    return days


def f6_path(step: str, n: int) -> list[Decimal]:
    return [Decimal(100) - Decimal(step) * t for t in range(1, n + 1)]


def sw(long_path: LongPath, x1: float, x2: float, hyst: ReleaseHyst) -> V08Switches:
    return V08Switches(x1=x1, x2=x2, t11_full=True, long_path=long_path, release_hyst=hyst)


F6_CASES = [  # (开关, 首次 START 的 t, 日期, 等级, 原因, 警戒的 t)
    ("V0", 40, date(2009, 7, 3), Level.WATCH, "R1_GE_K1", 80),
    ("X1_8_15", 80, date(2009, 8, 28), Level.WATCH, "R1L_CUMULATIVE", 150),
    ("X1_10_15", 100, date(2009, 9, 25), Level.WATCH, "R1L_CUMULATIVE", 150),
    ("X1_8_20", 80, date(2009, 8, 28), Level.WATCH, "R1L_CUMULATIVE", 200),
    ("X2_8_15", 150, date(2009, 12, 4), Level.ALERT, "R1L_CUMULATIVE", None),
    ("X2_8_20", 200, date(2010, 2, 12), Level.ALERT, "R1L_CUMULATIVE", None),
]


def f6_switches(name: str, hyst: ReleaseHyst) -> V08Switches:
    if name == "V0":
        return replace(V0, release_hyst=hyst)
    path, x1, x2 = name.split("_")
    return sw(LongPath(path), int(x1) / 100, int(x2) / 100, hyst)


@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("hyst", [ReleaseHyst.OLD, ReleaseHyst.EFFECTIVE])
@pytest.mark.parametrize(("name", "t_start", "day", "level", "reason", "t_alert"), F6_CASES)
def test_F6_uniform_decline(name, t_start, day, level, reason, t_alert, hyst, basis) -> None:  # type: ignore[no-untyped-def]
    run = f_run_decimal("t", f6_path("0.1", 300), params(f6_switches(name, hyst), basis=basis))
    assert run.labels["t0"] == date(2009, 5, 8)
    starts = [e for e in run.result.events if e.type is E.START]
    assert len(starts) == 1 and starts[0].day == run.labels[f"t{t_start}"] == day
    assert starts[0].basis["reasons"] == [reason]
    rec = run.rec(f"t{t_start}")
    assert rec.level is level and rec.p_ep == 100 and rec.atr_ep == 2
    assert rec.p_ep_day == run.labels["t0"] or name == "V0"
    escs = [e for e in run.result.events if e.type is E.ESCALATE]
    if t_alert is None:
        assert escs == []
    else:
        assert [e.day for e in escs] == [run.labels[f"t{t_alert}"]]
        assert escs[0].basis["reason"] == "D_EP_GE_K2"
    absent = {E.STABILIZED, E.RECOVERY_OBS, E.DOWNGRADE, E.RELEASE}
    assert not absent & {e.type for e in run.result.events}
    check_days(run, decline_days(t_start, "A" if level is Level.ALERT else "W", t_alert, 300))
    assert run.invariants() == []


@pytest.mark.parametrize("basis", BASES)
def test_F7_slower_decline(basis: AtrBasis) -> None:
    path = f6_path("0.03", 460)
    v0 = f_run_decimal("t", path, params(V0, basis=basis))
    assert not [e for e in v0.result.events if e.type is E.START]  # 460 日内不启动
    a = f_run_decimal("t", path, params(sw(LongPath.X1, 0.08, 0.15, ReleaseHyst.OLD), basis=basis))
    (start,) = [e for e in a.result.events if e.type is E.START]
    assert start.day == a.labels["t447"] == date(2011, 1, 25)
    assert start.basis["reasons"] == ["R1L_CUMULATIVE"]
    rec = a.rec("t447")
    assert rec.dd_l == pytest.approx(7.53 / 94.12) and rec.dd_l >= 0.08 - 1e-10
    assert (rec.p_ep, rec.p_ep_day, rec.atr_ep) == (
        pytest.approx(94.12),
        a.labels["t196"],
        pytest.approx(2.03),
    )
    assert rec.d_ep == pytest.approx(3.71, abs=0.01) and rec.level is Level.WATCH
    x10 = f_run_decimal(
        "t", path, params(sw(LongPath.X1, 0.10, 0.15, ReleaseHyst.OLD), basis=basis)
    )
    assert not [e for e in x10.result.events if e.type is E.START]
    check_days(v0, {})
    check_days(a, decline_days(447, "W", None, 460))
    check_days(x10, {})
    assert a.invariants() == [] and v0.invariants() == []
