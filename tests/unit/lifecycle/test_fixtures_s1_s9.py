"""12 号文档第 7.1 节 S1–S7、S9（S2 以 10 号第 7.1 节说明为准，S5 以扩展表为准；S8 本期不适用）。

共同前提：P_ep = 100、ATR_ep = 2；SMA10 为给定输入；TWO_STAGE、SMA（另列除外）。
D_ep = (100 − C) / 2；a ⇔ C > 92；a′ ⇔ C > 96；强恢复 ⇔ 确认成立且 C ≥ L_ep + 6。
"""

import pytest

from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.indicator_source import TableIndicators, TableRow
from stock_risk.domain.lifecycle.model import DataStatus, Display, EndReason, EventType, Level
from stock_risk.domain.lifecycle.params import ReleasePath
from stock_risk.domain.lifecycle.runner import run_with_source
from tests.unit.lifecycle.harness import START_DAY, SYMBOL, Run, params, s_run, weekdays

E = EventType
S1_ROWS = [
    ("s0", 80, None),
    ("s1", 78, None),
    ("s2", 79, 82),
    ("s3", 80, 81),
    ("s4", 81, 80),
    ("s5", 83, 80.5),
    ("s6", 85, 81),
    ("s7", 86, 81.5),
    ("s8", 87, 82),
    ("s9", 88, 82.5),
]


def assert_day(run: Run, label: str, **expected) -> None:
    rec = run.rec(label)
    for field, value in expected.items():
        got = getattr(rec, field)
        assert got == pytest.approx(value) if isinstance(value, float) else got == value, (
            f"{label}.{field}: 期望 {value!r}，实际 {got!r}"
        )


def test_S1_downgrade_then_release_while_d_ep_ge_k2() -> None:
    run = s_run(S1_ROWS, params())
    start = run.event("s0", E.START)
    assert start.basis["reasons"] == ["R1_GE_K2"] and start.transition == "T6"
    assert_day(run, "s0", l_ep=80.0, since_obs=0, d_ep=10.0, display=Display.ALERT, cycle_no=1)
    table = {  # 标签: (L_ep, since, 稳, 确, D_ep, U, 证据, dc 或 rc（日末）)
        "s1": (78, 0, False, False, 11, 0, set(), ("dc", 0)),
        "s2": (78, 1, False, False, 10.5, 0.5, set(), ("dc", 0)),
        "s3": (78, 2, False, False, 10, 1, set(), ("dc", 0)),
        "s4": (78, 3, True, True, 9.5, 1.5, {"b"}, ("dc", 1)),
        "s5": (78, 4, True, True, 8.5, 2.5, {"b"}, ("dc", 2)),
        "s6": (78, 5, True, True, 7.5, 3.5, {"b"}, ("dc", 0)),  # 3 → 0，降级
        "s7": (78, 6, True, True, 7, 4, {"b′"}, ("rc", 1)),
        "s8": (78, 7, True, True, 6.5, 4.5, {"b′"}, ("rc", 2)),
    }
    for label, (l_ep, since, stable, conf, d_ep, u, ev, (cnt, val)) in table.items():
        assert_day(
            run,
            label,
            l_ep=float(l_ep),
            since_obs=since,
            stable=stable,
            confirmed=conf,
            d_ep=float(d_ep),
            u=float(u),
            evidence=frozenset(ev),
        )
        assert getattr(run.rec(label), cnt) == val, label
    assert run.rec("s1").new_low and run.rec("s1").cycle_no == 2
    assert run.types("s4") == sorted([E.STABILIZED, E.RECOVERY_OBS])
    assert run.event("s4", E.RECOVERY_OBS).identity[1:] == (2, 1)  # 周期 2，第 1 次
    assert run.rec("s5").dc_cand == 2 and run.rec("s6").dc_cand == 3
    down = run.event("s6", E.DOWNGRADE)
    assert down.basis["reasons"] == ["b"] and down.transition == "T15"
    assert_day(run, "s6", level=Level.WATCH, display=Display.WATCH, downgraded=True, h_d=85.0)
    assert_day(run, "s7", giveback=0.0, h_d=86.0)  # 加重组未命中（回吐 0）
    assert run.rec("s9").rc_cand == 3
    rel = run.event("s9", E.RELEASE)
    assert rel.basis["reasons"] == ["b′"] and rel.transition == "T12"
    assert_day(run, "s9", display=Display.NORMAL, last_release=run.labels["s9"])
    for label in ("s1", "s2", "s3", "s5", "s7", "s8"):
        assert run.events(label) == [], label
    assert run.invariants() == []


def test_S2_no_reescalation_without_new_deterioration_and_rebuilt_observation() -> None:
    rows = [
        ("t0", 80, None),
        ("t1", 81, 83),
        ("t2", 82, 82.5),
        ("t3", 83, 82),
        ("t4", 84, 82.5),
        ("t5", 85, 83),
        ("t6", 84, 83.5),
        ("t7", 83, 83.8),
        ("t8", 82, 83.6),
        ("t9", 81, 83.2),
        ("t10", 81, 82.9),
        ("t11", 81, 82.6),
        ("t12", 81, 82.3),
    ]
    run = s_run(rows, params())
    exp = {  # since, 稳, 确, (H_d, 回吐), 证据, 计数
        "t1": (1, False, False, None, set(), ("dc", 0)),
        "t2": (2, False, False, None, set(), ("dc", 0)),
        "t3": (3, True, True, None, {"b"}, ("dc", 1)),
        "t4": (4, True, True, None, {"b"}, ("dc", 2)),
        "t5": (5, True, True, (85, None), {"b"}, ("dc", 0)),
        "t6": (6, True, True, (85, 0.5), set(), ("rc", 0)),
        "t7": (7, True, False, (85, 1.0), set(), ("rc", 0)),
        "t8": (8, True, False, (85, 1.5), set(), ("rc", 0)),
        "t9": (0, False, False, (None, 2.0), set(), ("dc", 0)),
        "t10": (1, False, False, None, set(), ("dc", 0)),
        "t11": (2, False, False, None, set(), ("dc", 0)),
        "t12": (3, True, False, None, set(), ("dc", 0)),
    }
    for label, (since, stable, conf, hd, ev, (cnt, val)) in exp.items():
        assert_day(
            run, label, since_obs=since, stable=stable, confirmed=conf, evidence=frozenset(ev)
        )
        assert getattr(run.rec(label), cnt) == val, label
        if hd is not None:
            h_d, gb = hd
            assert run.rec(label).h_d == (None if h_d is None else float(h_d)), label
            if gb is not None:
                assert run.rec(label).giveback == pytest.approx(gb), label
        elif label != "t5":
            assert run.rec(label).h_d is None, label
    assert run.types("t3") == sorted([E.STABILIZED, E.RECOVERY_OBS])
    assert run.types("t5") == [E.DOWNGRADE]
    assert run.event("t7", E.RECOVERY_REVOKED).basis["reason"] == "BELOW_SMA"
    esc = run.event("t9", E.ESCALATE)
    assert esc.transition == "T11" and esc.basis["reason"] == "GIVEBACK"
    # 10 号第 7.1 节说明：t9 无 STABILIZED、RECOVERY_OBS、RECOVERY_REVOKED、RECOVERY_FAIL
    assert run.types("t9") == [E.ESCALATE]
    assert_day(run, "t9", level=Level.ALERT, cycle_no=2, reset=True)
    assert run.types("t12") == [E.STABILIZED] and run.event("t12", E.STABILIZED).identity[1] == 2
    for label in ("t1", "t2", "t4", "t6", "t8", "t10", "t11"):
        assert run.events(label) == [], label
    assert run.invariants() == []


def test_S3_restricted_window_after_release() -> None:
    rows = [*S1_ROWS, ("s10", 87, None), ("s11", 86, None), ("s12", 85, None), ("s13", 84, None)]
    run = s_run(rows, params())
    for label, d_t in (("s10", 0.5), ("s11", 1.0), ("s12", 1.5)):
        assert_day(run, label, d_t=d_t, display=Display.NORMAL)
        assert run.events(label) == []
    assert_day(run, "s13", d_t=2.0, display=Display.WATCH, p_ep=88.0, p_ep_day=run.labels["s9"])
    start = run.event("s13", E.START)
    assert start.basis["reasons"] == ["R1_GE_K1"] and start.transition == "T7"
    assert run.invariants() == []


def test_S4_reason_switch_a_to_b_keeps_count() -> None:
    rows = [("u0", 90, None), ("u1", 93, 92), ("u2", 94, 92.2), ("u3", 92, 91.5)]
    run = s_run(rows, params())
    assert run.event("u0", E.START).basis["d_t"] == pytest.approx(5.0)
    for label, since, stable, conf, ev, dc in (
        ("u1", 1, False, False, {"a"}, 1),
        ("u2", 2, False, False, {"a"}, 2),
    ):
        assert_day(
            run,
            label,
            since_obs=since,
            stable=stable,
            confirmed=conf,
            evidence=frozenset(ev),
            dc=dc,
        )
    assert_day(
        run,
        "u3",
        since_obs=3,
        stable=True,
        confirmed=True,
        evidence=frozenset({"b"}),
        dc_cand=3,
        dc=0,
        level=Level.WATCH,
    )
    assert run.types("u3") == sorted([E.STABILIZED, E.RECOVERY_OBS, E.DOWNGRADE])
    down = run.event("u3", E.DOWNGRADE)
    assert down.basis["reasons"] == ["b"]
    assert down.basis["segment"] == [["a"], ["a"], ["b"]]
    assert run.invariants() == []


def test_S5_missing_days_cross_break_threshold() -> None:
    # 推导：v0 之前一日 vs 启动警戒（80），v0 继续警戒；v1–v5 缺价；v6 起为新历史段
    p = params()
    labels = ["p", "vs", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"]
    days = weekdays(START_DAY, len(labels))
    lab = dict(zip(labels, days, strict=True))
    cal = TradingCalendar.from_days(days)
    seg1 = TableIndicators(
        cal,
        {
            lab["p"]: TableRow(100, 100, atr=2),
            lab["vs"]: TableRow(100, 80, atr=2),
            lab["v0"]: TableRow(80, 81, atr=2),
        },
        n=20,
        prior_valid_bars=10_000,
    )
    seg2 = TableIndicators(
        cal,
        {lab["v6"]: TableRow(81, 81, atr=2), lab["v7"]: TableRow(81, 81)},
        n=20,
        prior_valid_bars=0,
    )  # 新历史段：指标从 v6 起重新积累
    first = run_with_source(SYMBOL, days[:8], seg1, p)
    second = run_with_source(SYMBOL, days[8:], seg2, p, state=first.final_state)
    run = Run(lab, type(first)(first.results + second.results, second.final_state, None), p)
    assert_day(run, "v0", display=Display.ALERT, m=0)
    for i, label in enumerate(("v1", "v2", "v3"), start=1):
        assert_day(
            run,
            label,
            m=i,
            display=Display.ALERT,
            data_status=DataStatus.MISSING,
            transition="T1",
            level=Level.ALERT,
        )
        assert run.types(label) == [E.DATA_ISSUE]
    eid = f"{SYMBOL}:{lab['vs']}"
    # 推导：断点日的诊断记录写明被结束的预警及终止原因
    assert_day(
        run,
        "v4",
        m=4,
        display=Display.DATA_BREAK,
        transition="T2",
        episode_id=eid,
        end_reason=EndReason.DATA_BREAK,
        level=None,
    )
    assert run.types("v4") == sorted([E.DATA_BREAK, E.DATA_ISSUE])
    assert run.event("v4", E.DATA_BREAK).identity == (eid, "DATA_BREAK", lab["v4"])
    assert_day(run, "v5", m=5, display=Display.DATA_BREAK, transition="T3")
    assert run.events("v5") == []
    assert_day(run, "v6", m=0, display=Display.INSUFFICIENT, transition="T4", seg_start=lab["v6"])
    assert_day(run, "v7", display=Display.INSUFFICIENT, transition="T5", seg_start=lab["v6"])
    assert run.events("v6") == [] and run.events("v7") == []
    assert run.rec("v4").last_release is None
    assert run.invariants() == []


@pytest.mark.parametrize("release", [ReleasePath.DIRECT, ReleasePath.TWO_STAGE])
def test_S6_strong_recovery_direct_vs_two_stage(release: ReleasePath) -> None:
    rows = [
        ("w0", 80, None),
        ("w1", 81, 83),
        ("w2", 82, 82.5),
        ("w3", 87, 82.8),
        ("w4", 88, 83.5),
        ("w5", 89, 84.2),
        ("w6", 90, 85),
        ("w7", 91, 85.8),
        ("w8", 92, 86.6),
    ]
    run = s_run(rows, params(release=release))
    direct = release is ReleasePath.DIRECT
    for label, since, conf, u, strong, dc, xc in (
        ("w1", 1, False, 0.5, False, 0, 0),
        ("w2", 2, False, 1.0, False, 0, 0),
        ("w3", 3, True, 3.5, True, 1, 1),
        ("w4", 4, True, 4.0, True, 2, 2),
    ):
        assert_day(
            run,
            label,
            since_obs=since,
            confirmed=conf,
            u=u,
            strong=strong,
            dc=dc,
            xc=xc if direct else 0,
        )
    assert run.types("w3") == sorted([E.STABILIZED, E.RECOVERY_OBS])
    assert run.rec("w5").dc_cand == 3
    if direct:
        assert run.rec("w5").xc_cand == 3
        rel = run.event("w5", E.RELEASE)
        assert rel.basis["reasons"] == ["DIRECT_STRONG_RECOVERY"] and rel.transition == "T14"
        for label in ("w6", "w7", "w8"):  # 受限窗口自 w5 起
            assert_day(run, label, display=Display.NORMAL, d_t=0.0)
            assert run.events(label) == []
    else:
        assert run.types("w5") == [E.DOWNGRADE]
        assert_day(run, "w6", rc=1, u=5.0, strong=True, evidence=frozenset({"b′"}))
        assert_day(run, "w7", rc=2, u=5.5)
        assert run.rec("w8").rc_cand == 3
        assert run.event("w8", E.RELEASE).basis["reasons"] == ["b′"]
        assert_day(run, "w8", display=Display.NORMAL)
    assert run.invariants() == []


def test_S7_repeated_confirmation_identities_and_rerun() -> None:
    rows = [
        ("x0", 80, None),
        ("x1", 81, 83),
        ("x2", 82, 82.5),
        ("x3", 83, 82),
        ("x4", 82, 82.5),
        ("x5", 84, 82.8),
    ]
    run = s_run(rows, params())
    eid = f"{SYMBOL}:{run.labels['x0']}"
    assert {e.key for e in run.events("x3")} == {
        (E.STABILIZED, eid, 1),
        (E.RECOVERY_OBS, eid, 1, 1),
    }
    assert {e.key for e in run.events("x4")} == {(E.RECOVERY_REVOKED, eid, 1, 1)}
    assert {e.key for e in run.events("x5")} == {(E.RECOVERY_OBS, eid, 1, 2)}
    # x5 重跑：同一输入再算一次，业务身份已存在，没有新业务事件
    from stock_risk.domain.lifecycle.engine import step

    x4_state = run.step("x4").state
    src_days = weekdays(START_DAY, 7)
    table = {
        d: TableRow(o, c, atr=2, sma=s)
        for d, o, c, s in zip(
            src_days,
            [100, 100, 80, 81, 82, 83, 82],
            [100, 80, 81, 82, 83, 82, 84],
            [None, None, 83, 82.5, 82, 82.5, 82.8],
            strict=True,
        )
    }
    src = TableIndicators(TradingCalendar.from_days(src_days), table, n=20, prior_valid_bars=10_000)
    rerun = step(x4_state, run.labels["x5"], src, run.params)
    seen = {e.key for r in run.result.results for e in r.events}
    assert {e.key for e in rerun.events} <= seen
    assert run.invariants() == []


def test_S9_atr_changes_and_peak_leaves_window() -> None:
    # 推导：y1–y70 每日收盘 85、SMA10 = 85；
    # 当日 ATR 按表在 y0 = 2、y3 = 4、y8 起 = 6，中间日线性取值
    labels = [f"y{i}" for i in range(71)]
    rows = [("y0", 84, None), *[(lab, 85, 85) for lab in labels[1:]]]
    atr = {"y1": 2.5, "y2": 3.0, "y3": 4.0, "y4": 4.5, "y5": 5.0, "y6": 5.5, "y7": 5.8}
    atr.update({f"y{i}": 6.0 for i in range(8, 71)})
    run = s_run(rows, params(), atr=atr)
    assert_day(run, "y0", since_obs=0, d_ep=8.0, display=Display.ALERT)
    assert_day(
        run, "y3", since_obs=3, stable=True, confirmed=False, long_stable=False, dc=0, d_ep=7.5
    )
    assert run.types("y3") == [E.STABILIZED]
    assert_day(
        run, "y8", since_obs=8, long_stable=True, confirmed=False, dc=1, evidence=frozenset({"c"})
    )
    assert_day(run, "y10", dc_cand=3, level=Level.WATCH)
    assert run.event("y10", E.DOWNGRADE).basis["reasons"] == ["c"]
    for i in range(11, 71):
        assert_day(run, f"y{i}", since_obs=i, rc=0, display=Display.WATCH, d_ep=7.5, atr_ep=2.0)
        assert run.events(f"y{i}") == []
    assert run.invariants() == []
