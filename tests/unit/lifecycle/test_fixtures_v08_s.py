"""13 号第 6 节 S23–S38（v0.8，10 号第 1A 节）。S37 情形 b 只作说明，不写测试。

共同前提（13 号第 6.0 节）：P_ep = 100、ATR_ep = 2、k₁ = 2、k₂ = 4、K = 3、d = 3、u_rel = 3、
φ = 0.5、h = 0.5、G = 3、K_stab = 8；TWO_STAGE、SMA（另行说明除外）。
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.indicator_source import ComputedIndicators, compute_r2
from stock_risk.domain.indicators import CalcStatus
from stock_risk.domain.lifecycle.model import Display, EventType, Level
from stock_risk.domain.lifecycle.params import (
    AtrBasis,
    IllegalCombination,
    LongPath,
    RecScale,
    V08Switches,
)
from stock_risk.domain.lifecycle.runner import run_bars
from stock_risk.domain.market_data import DailyBar
from stock_risk.domain.observation import DayPresence
from tests.unit.lifecycle import fields as fx
from tests.unit.lifecycle.fields import check_days, released, reset_at, started
from tests.unit.lifecycle.harness import FIXTURE_ASSUMPTIONS, START_DAY, Run, weekdays
from tests.unit.lifecycle.v08_harness import (
    V0,
    V2,
    A,
    B,
    Day,
    Prior,
    Table,
    params,
    prior_fields,
    run_table,
)

E = EventType
W, S, L = Level.WATCH, "S", "L"


def kinds(run, label):  # type: ignore[no-untyped-def]
    return sorted(e.type for e in run.events(label))


# ---------------- S23 T11 的完整警戒条件 ----------------


@pytest.mark.parametrize(
    ("case", "sw", "close", "escalates"),
    [("a", A, 70, False), ("b", A, 60, True), ("c", V0, 70, True)],
)
def test_S23_t11_full_alert_condition(case, sw, close, escalates) -> None:  # type: ignore[no-untyped-def]
    prior = Prior(W, l_ep=55, entry_path=S, delta_start=20, atr_ep=10, downgraded=True, h_d=90)
    run = run_table(Table([Day("t", close, sma=95)]), params(sw), prior)
    rec = run.rec("t")
    assert rec.giveback == pytest.approx((90 - close) / 10)
    assert rec.d_ep == pytest.approx((100 - close) / 10) and not rec.new_low
    if escalates:
        esc = run.event("t", E.ESCALATE)
        assert (esc.transition, esc.basis["reason"]) == ("T11", "GIVEBACK")
        assert rec.level is Level.ALERT and rec.reset and rec.h_d is None
        assert (rec.dc, rec.rc, rec.xc) == (0, 0, 0)
    else:
        assert run.events("t") == [] and rec.level is W
        assert rec.evidence == frozenset() and rec.rc == 0 and rec.h_d == 90
    # B-01 推导：加重时观察重置进入周期 2，H_d 清空；
    # 不加重时（情形 a）各字段同前态（无确认、无证据）
    check_days(
        run,
        {"t": {**reset_at("t", 2), "level": "A", "h_d": None, "events": [("ESC",)]}}
        if escalates
        else {},
        init=prior_fields(prior),
    )
    assert run.invariants() == []


def test_S24_t10_full_alert_condition() -> None:
    prior = Prior(W, l_ep=84, entry_path=S, delta_start=8, downgraded=True, h_d=90)
    run = run_table(Table([Day("t", 83.5, sma=95)]), params(A), prior)
    rec = run.rec("t")
    esc = run.event("t", E.ESCALATE)
    assert (esc.transition, esc.basis["reason"]) == ("T10", "NEW_LOW")
    assert rec.d_ep == pytest.approx(8.25) and rec.dd_ep == pytest.approx(0.165)
    assert rec.l_ep == 83.5 and rec.cycle_no == 2
    assert kinds(run, "t") == [E.ESCALATE]  # 不发 RECOVERY_FAIL、不发撤销
    # B-01 推导：新低与 T10 同日，只重置一次（周期 2），H_d 清空
    check_days(
        run,
        {"t": {**reset_at("t", 2), "level": "A", "h_d": None, "events": [("ESC",)]}},
        init=prior_fields(prior),
    )
    assert run.invariants() == []


# ---------------- S25 S 路径的百分比下限 ----------------


def s25_table(with_r1: bool) -> Table:
    days = [Day("r1", 96), Day("r2", 92), Day("r3", 88), Day("r4", 85)]
    return Table(days if with_r1 else days[1:], history=[Day("p", 100)])


def test_S25_s_path_percentage_floor_group_a() -> None:
    run = run_table(s25_table(True), params(A))
    r1 = run.rec("r1")
    assert (r1.d_t, r1.dd_t, r1.s_ok, r1.l_ok) == (
        pytest.approx(2.0),
        pytest.approx(0.04),
        False,
        False,
    )
    assert run.events("r1") == [] and r1.display is Display.NORMAL
    start = run.event("r2", E.START)
    assert start.basis["reasons"] == ["R1_GE_K1"] and start.transition == "T7"
    r2 = run.rec("r2")
    assert (r2.level, r2.p_ep, r2.atr_ep, r2.l_ep, r2.entry_path) == (W, 100, 2, 92, S)
    assert r2.d_t == pytest.approx(4.0) and r2.dd_t == pytest.approx(0.08)
    r3 = run.rec("r3")
    assert r3.new_low and r3.level is W and r3.reset and run.events("r3") == []
    esc = run.event("r4", E.ESCALATE)
    assert (esc.transition, esc.basis["reason"]) == ("T9", "D_EP_GE_K2")
    assert run.rec("r4").new_low and run.rec("r4").dd_ep == pytest.approx(0.15)
    # B-01 推导：r2 关注启动；r3 新低（周期 2）；r4 新低并 T9 加重（周期 3）
    check_days(
        run,
        {
            "r2": started("r2", "W"),
            "r3": reset_at("r3", 2),
            "r4": {**reset_at("r4", 3), "level": "A", "events": [("ESC",)]},
        },
    )
    assert run.invariants() == []


def test_S25_v0_contrast() -> None:
    run = run_table(s25_table(True), params(V0))
    assert run.event("r1", E.START).basis["reasons"] == ["R1_GE_K1"]
    alone = run_table(s25_table(False), params(V0))
    start = alone.event("r2", E.START)
    assert start.basis["reasons"] == ["R1_GE_K2"] and alone.rec("r2").level is Level.ALERT
    # B-01 推导：V0 下 r1 关注启动；r2 新低且 D_ep = 4，T9 加重（周期 2）；r3、r4 继续新低
    check_days(
        run,
        {
            "r1": started("r1", "W"),
            "r2": {**reset_at("r2", 2), "level": "A", "events": [("ESC",)]},
            "r3": reset_at("r3", 3),
            "r4": reset_at("r4", 4),
        },
    )
    # 只从 r2 起运行：r2 以警戒启动，r3、r4 新低
    check_days(alone, {"r2": started("r2", "A"), "r3": reset_at("r3", 2), "r4": reset_at("r4", 3)})


# ---------------- S26 R2 的百分比下限 ----------------

S26_SW = V08Switches(x1=0.08, x2=0.15, t11_full=True)


@pytest.mark.parametrize(
    ("case", "floor", "o", "c"), [("a", True, 93, 97), ("b", False, 93, 97), ("c", True, 88, 91)]
)
def test_S26_r2_percentage_floor(case, floor, o, c) -> None:  # type: ignore[no-untyped-def]
    sw = replace(S26_SW, r2_floor=floor)
    run = run_table(Table([Day("r1", c, open=o)], history=[Day("r0", 100)]), params(sw))
    rec = run.rec("r1")
    assert rec.r2_score == pytest.approx((o - 100) / 2) and rec.r2_triggered
    assert rec.d_t == pytest.approx((100 - c) / 2) and rec.dd_t == pytest.approx(1 - c / 100)
    if case == "a":
        assert run.events("r1") == [] and rec.r2_filtered is True
        check_days(run, {})  # B-01 推导：被过滤，不启动
        return
    start = run.event("r1", E.START)
    assert rec.level is W
    assert start.basis["reasons"] == (["GAP"] if case == "b" else ["R1_GE_K1", "GAP"])
    check_days(run, {"r1": started("r1", "W")})  # B-01 推导
    assert run.invariants() == []


# ---------------- S27 长窗口独立于短窗口，并受 last_release 截断 ----------------


def test_S27_long_window_independent_and_truncated_by_release() -> None:
    hist = [Day("old", 120)] + [Day(f"h{i}", 97) for i in range(99)]  # r 之前 100 日内曾有 120
    hist += [Day("r", 98)] + [Day(f"a{i}", 93) for i in range(1, 10)] + [Day("r10", 100)]
    hist += [Day(f"b{i}", 93) for i in range(11, 100)] + [Day("r100", 95)]
    hist += [Day(f"c{i}", 92) for i in range(101, 150)]
    run = run_table(Table([Day("T", 91)], history=hist), params(A), last_release_label="r")
    rec = run.rec("T")
    assert rec.dd_t == pytest.approx(1 - 91 / 95) and not rec.s_ok  # 短窗口最高 95
    assert rec.dd_l == pytest.approx(0.09) and rec.l_ok
    start = run.event("T", E.START)
    assert start.basis["reasons"] == ["R1L_CUMULATIVE"]
    assert (rec.p_ep, rec.p_ep_day, rec.atr_ep, rec.l_ep) == (100, run.labels["r10"], 2, 91)
    assert rec.d_ep == pytest.approx(4.5) and rec.dd_ep == pytest.approx(0.09)
    assert rec.level is W and rec.entry_path == L
    check_days(run, {"T": started("T", "W")}, init={"last_release": "r"})  # B-01 推导
    assert run.invariants() == []


# ---------------- S28 长窗口峰值日 ATR 不可得 ----------------


@pytest.mark.parametrize(
    ("case", "basis", "atr_t", "a_start", "selected", "d_ep"),
    [
        ("a", AtrBasis.PEAK, 2.5, 3.0, "PEAK_COMPONENT", 10 / 3),
        ("b", AtrBasis.MIN, 2.5, 2.5, "T", 4.0),
        ("c", AtrBasis.MIN, 3.5, 3.0, "PEAK_COMPONENT", 10 / 3),
    ],
)
def test_S28_long_peak_atr_unavailable(case, basis, atr_t, a_start, selected, d_ep) -> None:  # type: ignore[no-untyped-def]
    hist = [Day("s0", 100, atr=None)] + [Day(f"s{i}", 96, atr=None) for i in range(1, 20)]
    hist += [Day("q", 96, atr=3.0)] + [Day(f"s{i}", 96, atr=2.4) for i in range(21, 38)]
    hist += [Day(f"s{i}", 94, atr=2.4) for i in range(38, 60)] + [Day("s60", 95, atr=2.4)]
    hist += [Day(f"s{i}", 94, atr=2.4) for i in range(61, 100)]
    run = run_table(
        Table([Day("T", 90, atr=atr_t)], history=hist), params(A, basis=basis), prior_valid_bars=0
    )
    rec = run.rec("T")
    assert rec.dd_t == pytest.approx(1 - 90 / 95) and not rec.s_ok
    assert rec.dd_l == pytest.approx(0.10) and rec.l_ok
    v08 = run.event("T", E.START).basis["v08"]
    assert v08["p_ep_day"] == run.labels["s0"] and v08["peak_atr_available"] is False
    assert v08["peak_component_atr_day"] == run.labels["q"]
    assert v08["peak_component_atr"] == 3.0 and v08["atr_t"] == atr_t
    assert v08["a_start_selected"] == selected and v08["entry_path"] == L
    assert rec.atr_ep == a_start and rec.d_ep == pytest.approx(d_ep) and rec.level is W
    check_days(run, {"T": started("T", "W")})  # B-01 推导
    assert run.invariants() == []


# ---------------- S29 深跌后的反弹途中由 L 路径首次启动 ----------------


def test_S29_l_path_start_on_rebound_day() -> None:
    hist = [Day(f"x{i}", 90) for i in range(20)] + [Day("p", 100)]
    hist += [Day(f"y{i}", 90) for i in range(29)] + [Day("l", 80)]
    hist += [Day(f"z{i}", 85) for i in range(10)]
    hist += [Day(f"w{i}", 90) for i in range(30)] + [Day("pk", 92)]
    hist += [Day(f"v{i}", 90) for i in range(30)] + [Day("y", 86)]
    run = run_table(Table([Day("T", 88)], history=hist), params(A))
    rec = run.rec("T")
    assert rec.dd_t == pytest.approx(1 - 88 / 92) and not rec.s_ok  # 上涨日（前收 86）
    assert rec.dd_l == pytest.approx(0.12) and rec.l_ok
    assert run.event("T", E.START).basis["reasons"] == ["R1L_CUMULATIVE"]
    assert (rec.p_ep, rec.atr_ep, rec.level, rec.l_ep, rec.l_ep_day) == (
        100,
        2,
        W,
        80,
        run.labels["l"],
    )
    assert rec.d_ep == pytest.approx(6.0) and rec.dd_ep == pytest.approx(0.12)
    assert (88 - rec.l_ep) / rec.atr_ep == pytest.approx(4.0)  # U（推导：启动日不计算恢复进度）
    assert (rec.obs_start, rec.cycle_no) == (run.labels["T"], 1)
    check_days(run, {"T": started("T", "W")})  # B-01 推导
    assert run.invariants() == []


# ---------------- S30 禁用语义（V2） ----------------


@pytest.mark.parametrize(
    ("case", "close", "expect"),
    [
        ("a", 97, frozenset()),
        ("b", 98.2, frozenset({"a′"})),
        ("c", 91.5, frozenset()),
        ("d", 92.5, frozenset({"a"})),
    ],
)
def test_S30_disabled_semantics_v2(case, close, expect) -> None:  # type: ignore[no-untyped-def]
    if case in "ab":
        prior = Prior(W, l_ep=93, entry_path=S, delta_start=6)
    else:
        prior = Prior(Level.ALERT, l_ep=90, entry_path=S, delta_start=8)
    run = run_table(Table([Day("t", close, sma=99)]), params(V2), prior)
    rec = run.rec("t")
    assert rec.evidence == expect and rec.d_ep == pytest.approx((100 - close) / 2)
    if case in "ab":
        assert rec.theta == 4.0  # V2：E₁ = k₁ × ATR_ep = 4
    # B-01 推导：确认不成立（收盘低于 SMA10 = 99）；有证据时关注计 rc、警戒计 dc
    cnt = {"b": {"rc": 1}, "d": {"dc": 1}}.get(case)
    check_days(run, {"t": cnt} if cnt else {}, init=prior_fields(prior))
    assert run.invariants() == []


# ---------------- S31 a′ 的有效门槛回滞（B 组） ----------------


def test_S31a_percentage_dominates() -> None:
    prior = Prior(W, l_ep=90, entry_path=S, delta_start=8, atr_ep=1)
    days = [
        Day("1", 95.5, sma=99),
        Day("2", 96.5, sma=99),
        Day("3", 97, sma=99),
        Day("4", 97.5, sma=99),
    ]
    run = run_table(Table(days), params(B), prior)
    assert [run.rec(x).evidence for x in "123"] == [
        frozenset(),
        frozenset({"a′"}),
        frozenset({"a′"}),
    ]
    assert [run.rec(x).rc for x in "123"] == [0, 1, 2] and run.rec("1").theta == 8.0
    rel = run.event("4", E.RELEASE)
    assert rel.basis["reasons"] == ["a′"] and run.rec("4").rc_cand == 3
    old = run_table(Table(days), params(A), prior)  # release OLD：a′ 为 C > 98，四日都不成立
    assert all(old.rec(x).evidence == frozenset() for x in "1234")
    # B-01 推导：B 组 2、3 证据 a′，4 解除；OLD 下四日无证据
    check_days(run, {"2": {"rc": 1}, "3": {"rc": 2}, "4": released("4")}, init=prior_fields(prior))
    check_days(old, {}, init=prior_fields(prior))
    assert run.invariants() == [] and old.invariants() == []


@pytest.mark.parametrize(
    ("close", "b_ok", "old_ok"), [(95.5, True, True), (94.5, False, True), (92, False, True)]
)
def test_S31b_atr_dominates(close, b_ok, old_ok) -> None:  # type: ignore[no-untyped-def]
    prior = Prior(W, l_ep=90, entry_path=S, delta_start=10, atr_ep=5)
    for sw, ok in ((B, b_ok), (A, old_ok)):
        run = run_table(Table([Day("t", close, sma=99)]), params(sw), prior)
        assert (run.rec("t").evidence == frozenset({"a′"})) is ok
        if sw is B:
            assert run.rec("t").theta == 10.0
        # B-01 推导：a′ 成立时 rc 计 1，其余字段同前态
        check_days(run, {"t": {"rc": 1}} if ok else {}, init=prior_fields(prior))


# ---------------- S32 b′ 绕过回滞后解除（B 组） ----------------


def test_S32_strong_recovery_bypasses_hysteresis() -> None:
    prior = Prior(W, l_ep=80, entry_path=S, delta_start=8, downgraded=True, h_d=87, stabilized=True)
    days = [Day("1", 88, sma=85), Day("2", 88.5, sma=85.5), Day("3", 89, sma=86)]
    run = run_table(Table(days), params(B), prior)
    for x, hd, rc in (("1", 88, 1), ("2", 88.5, 2)):
        rec = run.rec(x)
        assert (rec.confirmed, rec.strong, rec.evidence, rec.rc, rec.h_d) == (
            True,
            True,
            frozenset({"b′"}),
            rc,
            hd,
        )
        assert rec.giveback == 0 and rec.strong_threshold == 86
    assert kinds(run, "1") == [E.RECOVERY_OBS] and run.event("1", E.RECOVERY_OBS).identity[2] == 1
    assert run.events("2") == []
    assert kinds(run, "3") == [E.RELEASE] and run.event("3", E.RELEASE).basis["reasons"] == ["b′"]
    # B-01 推导：1 确认（第 1 次）且强恢复，H_d 随收盘上移；3 解除
    check_days(
        run,
        {
            "1": {
                "prev_confirm": True,
                "obs_issued": True,
                "h_d": 88.0,
                "rc": 1,
                "events": [("OBS", 1, 1)],
            },
            "2": {"h_d": 88.5, "rc": 2},
            "3": released("3"),
        },
        init=prior_fields(prior),
    )
    assert run.invariants() == []


# ---------------- S33 恢复尺度缩放后解除、随后再跌（B 组） ----------------


PRIOR_S33 = Prior(
    W, l_ep=10, entry_path=S, delta_start=20, atr_ep=6, downgraded=True, h_d=11.5, stabilized=True
)


def s33_run(rec_scale: RecScale):  # type: ignore[no-untyped-def]
    prior = PRIOR_S33
    days = [
        Day("1", 12.0, sma=11.0, atr=6),
        Day("2", 12.2, sma=11.2, atr=6),
        Day("3", 12.4, sma=11.4, atr=0.6),
        Day("4", 11.6, sma=11.6, atr=0.6),
        Day("5", 11.2, sma=11.6, atr=0.6),
    ]
    return run_table(Table(days), params(replace(B, rec_scale=rec_scale)), prior)


def test_S33_price_scale_release_then_restart() -> None:
    run = s33_run(RecScale.PRICE)
    r1 = run.rec("1")
    assert r1.atr_rec == pytest.approx(0.6) and r1.strong_threshold == pytest.approx(11.8)
    assert r1.evidence == frozenset({"b′"}) and r1.rc == 1 and r1.theta == 12
    assert kinds(run, "1") == [E.RECOVERY_OBS]
    assert run.rec("2").rc == 2 and run.rec("2").h_d == pytest.approx(12.2)
    assert run.event("3", E.RELEASE).basis["reasons"] == ["b′"]
    assert run.rec("3").last_release == run.labels["3"]
    r4 = run.rec("4")
    assert (r4.d_t, r4.dd_t) == (pytest.approx(0.8 / 0.6), pytest.approx(1 - 11.6 / 12.4))
    assert run.events("4") == [] and r4.display is Display.NORMAL
    start = run.event("5", E.START)
    assert start.basis["reasons"] == ["R1_GE_K1"]
    r5 = run.rec("5")
    assert r5.d_t == pytest.approx(2.0) and r5.dd_t == pytest.approx(1 - 11.2 / 12.4)
    assert (r5.p_ep, r5.p_ep_day, r5.atr_ep, r5.l_ep) == (12.4, run.labels["3"], 0.6, 11.2)
    assert r5.episode_id != run.rec("1").episode_id
    # B-01 推导：1 确认且强恢复；3 解除；4 无预警（last_release = 3）；5 以关注重新启动
    check_days(
        run,
        {
            "1": {
                "prev_confirm": True,
                "obs_issued": True,
                "h_d": 12.0,
                "rc": 1,
                "events": [("OBS", 1, 1)],
            },
            "2": {"h_d": 12.2, "rc": 2},
            "3": released("3"),
            "5": started("5", "W"),
        },
        init=prior_fields(PRIOR_S33),
    )
    assert run.invariants() == []


def test_S33_off_scale_no_release() -> None:
    run = s33_run(RecScale.OFF)
    r1 = run.rec("1")
    assert r1.strong_threshold == 28 and r1.confirmed and not r1.strong
    assert r1.evidence == frozenset() and kinds(run, "1") == [E.RECOVERY_OBS]
    assert run.rec("2").evidence == frozenset() and run.rec("3").evidence == frozenset()
    assert run.rec("3").level is W and all(e.type is not E.RELEASE for e in run.result.events)
    # B-01 推导：1 确认（无强恢复，rc 0）；H_d 随收盘升至 12.4；4 收盘等于 SMA10，确认消失并撤销
    check_days(
        run,
        {
            "1": {"prev_confirm": True, "obs_issued": True, "h_d": 12.0, "events": [("OBS", 1, 1)]},
            "2": {"h_d": 12.2},
            "3": {"h_d": 12.4},
            "4": {"prev_confirm": False, "events": [("REV", 1, 1)]},
        },
        init=prior_fields(PRIOR_S33),
    )


# ---------------- S34 I21：证据交替的解除原因（V0） ----------------


def test_S34_i21_alternating_evidence_release_reason() -> None:
    prior = Prior(W, l_ep=88, entry_path=S, delta_start=6, downgraded=True, h_d=96, stabilized=True)
    days = [Day("1", 96.5, sma=97), Day("2", 95.5, sma=95), Day("3", 96.4, sma=97)]
    run = run_table(Table(days), params(V0), prior)
    exp = [(False, False, {"a′"}, 1), (True, True, {"b′"}, 2)]
    for x, (conf, strong, ev, rc) in zip("12", exp, strict=True):
        rec = run.rec(x)
        assert (rec.confirmed, rec.strong, rec.evidence, rec.rc) == (
            conf,
            strong,
            frozenset(ev),
            rc,
        )
    assert run.rec("1").d_ep == pytest.approx(1.75) and run.rec("2").d_ep == pytest.approx(2.25)
    assert kinds(run, "2") == [E.RECOVERY_OBS]
    rel = run.event("3", E.RELEASE)
    assert rel.basis["reasons"] == ["a′"] and rel.basis["segment"] == [["a′"], ["b′"], ["a′"]]
    assert kinds(run, "3") == [E.RELEASE]  # 当日不发 RECOVERY_REVOKED
    # B-01 推导：1 只有 a′，H_d 升至 96.5；2 确认（第 1 次）且强恢复；3 解除，当日不撤销
    check_days(
        run,
        {
            "1": {"h_d": 96.5, "rc": 1},
            "2": {"prev_confirm": True, "obs_issued": True, "rc": 2, "events": [("OBS", 1, 1)]},
            "3": released("3"),
        },
        init=prior_fields(prior),
    )
    assert run.invariants() == []


# ---------------- S35 I18：跨缺价的 TR 与断段后重新预热 ----------------


def test_S35_tr_across_missing_day_and_rewarm_after_break() -> None:
    days = weekdays(START_DAY, 3)
    cal = TradingCalendar.from_days(days)
    a = DailyBar(days[0], 100, 101, 99, 100, 1, 100, 101, 99, 100)
    c = DailyBar(days[2], 91, 92, 89, 90, 1, 91, 92, 89, 90)
    src = ComputedIndicators(cal, [a, c])
    from stock_risk.domain.indicators import true_range

    assert true_range([a, c], 1) == 11  # 前收取日 a
    assert compute_r2(src, days[2], 1, 3).status is CalcStatus.MISSING_DATA
    # 连续缺价 G + 1 = 4 日后新段：第 82 条不可计算、第 83 条可计算；ATR₂₀ 自第 21 条起可得
    from tests.unit.lifecycle.harness import f_bars

    flat = f_bars([100.0] * 10)  # 100 根平盘日线
    allbars = flat[:10] + flat[14 : 14 + 83]
    cal2 = TradingCalendar.from_days(b.day for b in flat[: 14 + 83])
    presence = [DayPresence(d, d in {b.day for b in allbars}) for d in cal2.days]
    out = run_bars("X", cal2, presence, allbars, params(V0))
    seg2 = [r.record for r in out.results if r.record.day >= flat[14].day]
    assert seg2[0].seg_start == flat[14].day and seg2[0].transition == "T4"
    assert seg2[81].display is Display.INSUFFICIENT and seg2[82].display is Display.NORMAL
    seg_src = ComputedIndicators(cal2, flat[14 : 14 + 83])
    assert seg_src.atr(flat[14 + 19].day, 20) is None and seg_src.atr(flat[14 + 20].day, 20) == 2
    # B-01 推导：平盘，各日无预警；4 个缺价日（第 4 日为断点，无预警故无 DATA_BREAK）只有 DATA_ISSUE
    gap = {f"m{i}": flat[10 + i].day for i in range(4)}
    check_days(Run(gap, out, params(V0)), {k: {"events": [("ISSUE",)]} for k in gap}, symbol="X")


# ---------------- S36 非法组合 ----------------


@pytest.mark.parametrize(
    ("kwargs", "constraint"),
    [
        ({"x2": 0.15, "t11_full": False}, "约束 1"),
        ({"long_path": LongPath.X1}, "约束 2"),
        ({"long_path": LongPath.X2}, "约束 2"),
        ({"r2_floor": True}, "约束 3"),
        ({"x1": 0.10, "x2": 0.10, "t11_full": True}, "约束 4"),
    ],
)
def test_S36_illegal_switch_combinations(kwargs, constraint) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(IllegalCombination, match=constraint):
        V08Switches(**kwargs)


@pytest.mark.parametrize("h", [0, 1.2])
def test_S36_illegal_h(h: float) -> None:
    with pytest.raises(IllegalCombination, match="约束 5"):
        replace(FIXTURE_ASSUMPTIONS, h=h)


# ---------------- S37 L 路径启动后不涨不解除 ----------------


@pytest.mark.parametrize(("case", "sw", "released"), [("a", B, False), ("c", A, True)])
def test_S37_l_path_no_release_without_rebound(case, sw, released) -> None:  # type: ignore[no-untyped-def]
    hist = [Day(f"x{i}", 90, atr=10) for i in range(20)] + [Day("p", 100, atr=10)]
    hist += [Day(f"y{i}", 94, atr=10) for i in range(69)] + [Day("k", 95, atr=10)]
    hist += [Day(f"z{i}", 94, atr=10) for i in range(29)]
    days = [Day("0", 92, atr=10)] + [Day(str(i), 92, sma=95, atr=10) for i in (1, 2, 3)]
    run = run_table(Table(days, history=hist), params(sw))
    r0 = run.rec("0")
    assert r0.dd_t == pytest.approx(1 - 92 / 95) and r0.dd_l == pytest.approx(0.08)
    assert run.event("0", E.START).basis["reasons"] == ["R1L_CUMULATIVE"]
    assert (r0.p_ep, r0.atr_ep, r0.entry_path, r0.delta_start) == (100, 10, L, 8)
    assert r0.d_ep == pytest.approx(0.8)
    if released:
        assert run.event("3", E.RELEASE).basis["reasons"] == ["a′"]
        assert [run.rec(x).rc for x in "12"] == [1, 2]
    else:
        assert run.rec("1").theta == 8.0
        assert all(run.rec(x).rc == 0 for x in "123") and run.rec("3").level is W
    # B-01 推导：0 关注启动；B 组 a′ 不成立，3 为 since_obs = 3，STABILIZED（周期 1）；
    # A 组（OLD）a′ 成立，1、2 计数，3 解除（解除日不发 STABILIZED）
    if released:
        check_days(
            run, {"0": started("0", "W"), "1": {"rc": 1}, "2": {"rc": 2}, "3": fx.released("3")}
        )
    else:
        check_days(run, {"0": started("0", "W"), "3": {"events": [("STAB", 1)]}})
    assert run.invariants() == []


# ---------------- S38 R 路径启动后不涨不解除 ----------------

S38_SW = V08Switches(x1=0.08, x2=0.15, t11_full=True, release_hyst=B.release_hyst)


@pytest.mark.parametrize(("sw", "released"), [(S38_SW, False), (V0, True)])
def test_S38_r_path_no_release_without_rebound(sw, released) -> None:  # type: ignore[no-untyped-def]
    days = [Day("r1", 97, open=93)] + [Day(f"r{i}", 97, sma=99) for i in (2, 3, 4)]
    run = run_table(Table(days, history=[Day("r0", 100)]), params(sw))
    r1 = run.rec("r1")
    assert run.event("r1", E.START).basis["reasons"] == ["GAP"] and r1.level is W
    assert (r1.entry_path, r1.delta_start) == ("R", 3)
    if released:
        assert [run.rec(x).rc for x in ("r2", "r3")] == [1, 2]
        assert run.event("r4", E.RELEASE).basis["reasons"] == ["a′"]
    else:
        assert run.rec("r2").theta == 3.0
        assert all(run.rec(x).rc == 0 for x in ("r2", "r3", "r4"))
        assert run.rec("r4").level is W
    # B-01 推导：同 S37；r4 为 since_obs = 3
    if released:
        check_days(
            run,
            {"r1": started("r1", "W"), "r2": {"rc": 1}, "r3": {"rc": 2}, "r4": fx.released("r4")},
        )
    else:
        check_days(run, {"r1": started("r1", "W"), "r4": {"events": [("STAB", 1)]}})
    assert run.invariants() == []
