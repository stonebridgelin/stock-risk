"""10 号第 7.2 节 S10–S13、S15；13 号第 3 节 S16–S19、S21、S22（S14、S20 在 F 组文件中）。"""

from pathlib import Path

import pytest

from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.indicator_source import ComputedIndicators, TableIndicators, TableRow
from stock_risk.domain.lifecycle.model import DataStatus, Display, EventType, Level
from stock_risk.domain.lifecycle.params import FIXED, AtrBasis, ReleasePath
from stock_risk.domain.lifecycle.runner import run_with_source
from stock_risk.domain.numeric import quantize_price
from tests.unit.lifecycle.fields import check_days, released, reset_at, started
from tests.unit.lifecycle.harness import START_DAY, SYMBOL, Run, f_bars, params, s_run, weekdays
from tests.unit.lifecycle.test_fixtures_s1_s9 import S1_DAYS, S1_ROWS, assert_day

E = EventType
DIRECT, TWO = ReleasePath.DIRECT, ReleasePath.TWO_STAGE


@pytest.mark.parametrize("path", ["甲", "乙"])
def test_S10_direct_release_monotonicity(path: str) -> None:
    d4 = 83 if path == "甲" else 82.4
    rows = [
        ("d1", 80, None),
        ("d2", 81, 84),
        ("d3", 82, 83),
        ("d4", d4, 82.5),
        ("d5", 87, 83),
        ("d6", 89, 83.5),
        ("d7", 90, 84),
    ]
    run = s_run(rows, params(release=DIRECT))
    exp = {  # 确, 强, dc（日末）, xc（日末）
        "甲": {
            "d2": (False, False, 0, 0),
            "d3": (False, False, 0, 0),
            "d4": (True, False, 1, 0),
            "d5": (True, True, 2, 1),
            "d6": (True, True, 0, 2),
        },
        "乙": {
            "d2": (False, False, 0, 0),
            "d3": (False, False, 0, 0),
            "d4": (False, False, 0, 0),
            "d5": (True, True, 1, 1),
            "d6": (True, True, 2, 2),
        },
    }[path]
    for label, (conf, strong, dc, xc) in exp.items():
        assert_day(run, label, confirmed=conf, strong=strong, dc=dc, xc=xc)
    if path == "甲":
        assert run.types("d4") == sorted([E.STABILIZED, E.RECOVERY_OBS])
        assert run.types("d6") == [E.DOWNGRADE] and run.rec("d6").level is Level.WATCH
        assert run.rec("d6").dc_cand == 3
        rel = run.event("d7", E.RELEASE)
        assert rel.transition == "T12" and run.rec("d7").xc_cand == 3
        # E-03：只有 xc 达到 d，原因只记 DIRECT；当日证据 {b′} 另存
        assert rel.basis["reasons"] == ["DIRECT_STRONG_RECOVERY"]
        assert rel.basis["today_evidence"] == ["b′"]
    else:
        assert run.types("d4") == [E.STABILIZED] and run.types("d5") == [E.RECOVERY_OBS]
        rel = run.event("d7", E.RELEASE)
        assert rel.transition == "T14" and rel.basis["reasons"] == ["DIRECT_STRONG_RECOVERY"]
        assert run.rec("d7").dc_cand == 3 and run.rec("d7").xc_cand == 3
    assert run.rec("d7").display is Display.NORMAL
    # B-01 推导：甲 d4 确认（dc 1），d5 强恢复起 xc 计数，d6 降级（H_d = 89，xc 保留 2），
    # d7 xc 达到 3 解除；
    # 乙 d4 只企稳，d5 确认且强恢复，d7 dc 与 xc 同时达到 3，T14 解除
    if path == "甲":
        check_days(
            run,
            {
                "d1": started("d1", "A"),
                "d4": {
                    "prev_confirm": True,
                    "obs_issued": True,
                    "dc": 1,
                    "events": [("STAB", 1), ("OBS", 1, 1)],
                },
                "d5": {"dc": 2, "xc": 1},
                "d6": {
                    "level": "W",
                    "downgraded": True,
                    "h_d": 89.0,
                    "dc": 0,
                    "rc": 0,
                    "xc": 2,
                    "events": [("DOWN",)],
                },
                "d7": released("d7"),
            },
        )
    else:
        check_days(
            run,
            {
                "d1": started("d1", "A"),
                "d4": {"events": [("STAB", 1)]},
                "d5": {
                    "prev_confirm": True,
                    "obs_issued": True,
                    "dc": 1,
                    "xc": 1,
                    "events": [("OBS", 1, 1)],
                },
                "d6": {"dc": 2, "xc": 2},
                "d7": released("d7"),
            },
        )
    assert run.invariants() == []


def test_S11_shallow_drop_release_by_half_recovery() -> None:
    rows = [
        ("d1", 86, None),
        ("d2", 90, 92),
        ("d3", 92, 91.5),
        ("d4", 93, 91),
        ("d5", 93.5, 92),
        ("d6", 93.5, 92.5),
        ("d7", 93.5, 93),
        ("d8", 93.5, 93.2),
    ]
    run = s_run(rows, params(k1=3, k2=5, u_rel=5))
    assert run.event("d1", E.START).basis["d_t"] == pytest.approx(7.0)
    assert run.rec("d1").level is Level.ALERT
    for label, since, conf, strong, ev, cnt, val in (
        ("d2", 1, False, False, set(), "dc", 0),
        ("d3", 2, False, False, {"a"}, "dc", 1),
        ("d4", 3, True, True, {"a", "b"}, "dc", 2),
        ("d6", 5, True, True, {"b′"}, "rc", 1),
        ("d7", 6, True, True, {"b′"}, "rc", 2),
    ):
        assert_day(
            run, label, since_obs=since, confirmed=conf, strong=strong, evidence=frozenset(ev)
        )
        assert getattr(run.rec(label), cnt) == val, label
    assert run.rec("d4").strong_threshold == pytest.approx(93.0)  # L_ep 86 + min(10, 7)
    assert run.types("d4") == sorted([E.STABILIZED, E.RECOVERY_OBS])
    assert run.rec("d5").dc_cand == 3
    assert run.event("d5", E.DOWNGRADE).basis["reasons"] == ["a", "b"]
    assert run.rec("d8").rc_cand == 3
    assert run.event("d8", E.RELEASE).basis["reasons"] == ["b′"]
    # B-01 推导：d3 起证据 a，d4 确认；d5 降级（H_d = 93.5，此后收盘不高于 93.5）；d6–d8 b′，d8 解除
    check_days(
        run,
        {
            "d1": started("d1", "A"),
            "d3": {"dc": 1},
            "d4": {
                "prev_confirm": True,
                "obs_issued": True,
                "dc": 2,
                "events": [("STAB", 1), ("OBS", 1, 1)],
            },
            "d5": {
                "level": "W",
                "downgraded": True,
                "h_d": 93.5,
                "dc": 0,
                "rc": 0,
                "events": [("DOWN",)],
            },
            "d6": {"rc": 1},
            "d7": {"rc": 2},
            "d8": released("d8"),
        },
    )
    assert run.invariants() == []


@pytest.mark.parametrize("basis", [AtrBasis.PEAK, AtrBasis.MIN])
def test_S12_start_denominator_peak_vs_min(basis: AtrBasis) -> None:
    # 推导：d2–d39 取与 d40 相同的收盘 99.8、ATR 1.0（表中未列出，取值使其不触发启动）
    rows = [
        ("d1", 100, None),
        *[(f"d{i}", 99.8, None) for i in range(2, 41)],
        ("d41", 96, None),
        ("d42", 92, None),
        ("d43", 88.5, None),
    ]
    atr = {"d1": 6.0, **{f"d{i}": 1.0 for i in range(2, 41)}, "d41": 1.2, "d42": 1.5, "d43": 1.8}
    run = s_run(rows, params(basis=basis), atr=atr, peak_atr=1.0)
    if basis is AtrBasis.PEAK:
        for label, d_t in (("d40", 0.2 / 6), ("d41", 4 / 6), ("d42", 8 / 6), ("d43", 11.5 / 6)):
            assert_day(run, label, d_t=d_t, a_start=6.0, display=Display.NORMAL)
        assert all(not r.events for r in run.result.results)  # 跌 11.5% 未报警
    else:
        assert_day(run, "d40", a_start=1.0, d_t=0.2 / 1.0, display=Display.NORMAL)
        assert_day(run, "d41", a_start=1.2, d_t=4 / 1.2, atr_ep=1.2, level=Level.WATCH)
        assert run.event("d41", E.START).basis["reasons"] == ["R1_GE_K1"]
        assert_day(run, "d42", d_ep=8 / 1.2, level=Level.ALERT)
        esc = run.event("d42", E.ESCALATE)
        assert esc.transition == "T9" and esc.basis["reason"] == "D_EP_GE_K2"
        assert_day(run, "d43", d_ep=11.5 / 1.2, level=Level.ALERT)
    # B-01 推导：PEAK 全程无预警；MIN 在 d41 以关注启动，
    # d42 新低且 T9 加重（同日只重置一次，周期 2），
    # d43 再创新低（周期 3）；从未发出恢复观察，不发 RECOVERY_FAIL
    if basis is AtrBasis.PEAK:
        check_days(run, {})
    else:
        check_days(
            run,
            {
                "d41": started("d41", "W"),
                "d42": {**reset_at("d42", 2), "level": "A", "events": [("ESC",)]},
                "d43": reset_at("d43", 3),
            },
        )
    assert run.invariants() == []


@pytest.mark.parametrize(("basis", "starts"), [(AtrBasis.PEAK, False), (AtrBasis.MIN, True)])
def test_S12_cost_example(basis: AtrBasis, starts: bool) -> None:
    # 10 号 S12 代价示例：峰值 100（ATR 2），之后 ATR 0.5，收盘 98.9：
    # MIN 的 D = 2.2，PEAK 的 D = 0.55
    run = s_run([("q1", 98.9, None)], params(basis=basis), atr={"q1": 0.5})
    assert_day(run, "q1", d_t=2.2 if starts else 0.55)
    assert (run.types("q1") == [E.START]) is starts
    # B-01 推导：D = 2.2 时以关注启动（2 ≤ D < 4）
    check_days(run, {"q1": started("q1", "W")} if starts else {})


def test_S13_progress_on_escalation_day() -> None:
    rows = [
        ("d1", 80, None),
        ("d2", 85, 88),
        ("d3", 86, 86),
        ("d4", 87, 85),
        ("d5", 88, 84),
        ("d6", 90, 83),
        ("d7", 86, 82),
        ("d8", 86, 82),
        ("d9", 86, 82),
        ("d10", 86, 82),
    ]
    run = s_run(rows, params())
    eid = f"{SYMBOL}:{run.labels['d1']}"
    for label, since, cycle, conf in (
        ("d2", 1, 1, False),
        ("d3", 2, 1, False),
        ("d4", 3, 1, True),
        ("d5", 4, 1, True),
        ("d6", 5, 1, True),
        ("d7", 0, 2, False),
        ("d8", 1, 2, False),
        ("d9", 2, 2, False),
        ("d10", 3, 2, True),
    ):
        assert_day(run, label, since_obs=since, cycle_no=cycle, confirmed=conf)
    assert {e.key for e in run.events("d4")} == {
        (E.STABILIZED, eid, 1),
        (E.RECOVERY_OBS, eid, 1, 1),
    }
    assert_day(run, "d4", dc=1)
    assert_day(run, "d5", dc=2)
    assert run.types("d6") == [E.DOWNGRADE] and run.rec("d6").h_d == 90.0
    assert run.rec("d6").giveback is None  # 降级当日仍为警戒，第 6 步不计算回吐
    assert_day(run, "d7", giveback=2.0, level=Level.ALERT, h_d=None)
    assert run.event("d7", E.ESCALATE).basis["reason"] == "GIVEBACK"
    revoked = run.event("d7", E.RECOVERY_REVOKED)
    assert revoked.basis["reason"] == "ESCALATED" and revoked.identity == (eid, 1, 1)
    assert run.types("d7") == sorted([E.ESCALATE, E.RECOVERY_REVOKED])
    assert run.events("d8") == [] and run.events("d9") == []
    assert {e.key for e in run.events("d10")} == {
        (E.STABILIZED, eid, 2),
        (E.RECOVERY_OBS, eid, 2, 1),
    }
    assert_day(run, "d10", dc=1)
    # B-01 推导：d4 确认，d6 降级（H_d = 90）；d7 回吐 2，T11 加重：观察重置进入周期 2，H_d 清空，
    # 撤销周期 1 的确认（ESCALATED）；d10 周期 2 企稳并确认
    check_days(
        run,
        {
            "d1": started("d1", "A"),
            "d4": {
                "prev_confirm": True,
                "obs_issued": True,
                "dc": 1,
                "events": [("STAB", 1), ("OBS", 1, 1)],
            },
            "d5": {"dc": 2},
            "d6": {
                "level": "W",
                "downgraded": True,
                "h_d": 90.0,
                "dc": 0,
                "rc": 0,
                "events": [("DOWN",)],
            },
            "d7": {
                **reset_at("d7", 2),
                "level": "A",
                "h_d": None,
                "events": [("ESC",), ("REV", 1, 1)],
            },
            "d10": {
                "prev_confirm": True,
                "obs_issued": True,
                "dc": 1,
                "events": [("STAB", 2), ("OBS", 2, 1)],
            },
        },
    )
    assert run.invariants() == []


def test_S15_release_from_watch() -> None:
    rows = [
        ("d1", 96, None),
        ("d2", 94, None),
        ("d3", 95, 96),
        ("d4", 95.5, 96),
        ("d5", 97, 96),
        ("d6", 97.5, 96.3),
        ("d7", 98, 96.6),
    ]
    run = s_run(rows, params())
    assert run.event("d1", E.START).basis["d_t"] == pytest.approx(2.0)
    assert run.rec("d1").level is Level.WATCH
    assert_day(
        run,
        "d2",
        l_ep=94.0,
        since_obs=0,
        new_low=True,
        d_ep=3.0,
        level=Level.WATCH,
        rc=0,
        cycle_no=2,
    )  # D_ep = 3 < 4，不加重
    assert run.events("d2") == []
    for label, since, conf, strong, ev, rc in (
        ("d3", 1, False, False, set(), 0),
        ("d4", 2, False, False, set(), 0),
        ("d5", 3, True, True, {"a′", "b′"}, 1),
        ("d6", 4, True, True, {"a′", "b′"}, 2),
    ):
        assert_day(
            run,
            label,
            since_obs=since,
            confirmed=conf,
            strong=strong,
            evidence=frozenset(ev),
            rc=rc,
            l_ep=94.0,
        )
    assert run.types("d5") == sorted([E.STABILIZED, E.RECOVERY_OBS])
    assert run.rec("d7").rc_cand == 3
    rel = run.event("d7", E.RELEASE)
    assert rel.basis["reasons"] == ["a′", "b′"]  # I21：d7 当日证据 {a′, b′}
    assert run.rec("d7").display is Display.NORMAL
    # B-01 推导：d2 新低（周期 2），不加重；d5 确认，rc 自 d5 起计数；未降级，H_d 始终为空
    check_days(
        run,
        {
            "d1": started("d1", "W"),
            "d2": reset_at("d2", 2),
            "d5": {
                "prev_confirm": True,
                "obs_issued": True,
                "rc": 1,
                "events": [("STAB", 2), ("OBS", 2, 1)],
            },
            "d6": {"rc": 2},
            "d7": released("d7"),
        },
    )
    assert run.invariants() == []


def test_S16_interrupted_strong_recovery_does_not_release() -> None:
    rows = [
        ("d1", 80, None),
        ("d2", 81, 84),
        ("d3", 82, 83),
        ("d4", 83, 82.5),
        ("d5", 87, 83),
        ("d6", 89, 83.5),
        ("d7", 88, 89),
        ("d8", 91, 89.5),
        ("d9", 92, 90),
        ("d10", 93, 90.5),
    ]
    run = s_run(rows, params(release=DIRECT))
    eid = f"{SYMBOL}:{run.labels['d1']}"
    # 表：判定（今日候选）dc / rc / xc 与日末 dc / rc / xc；"—"为不适用（推导：日末保持 0）
    exp = {
        "d2": ((0, None, 0), (0, 0, 0), False, False, set()),
        "d3": ((0, None, 0), (0, 0, 0), False, False, set()),
        "d4": ((1, None, 0), (1, 0, 0), True, False, {"b"}),
        "d5": ((2, None, 1), (2, 0, 1), True, True, {"b"}),
        "d6": ((3, None, 2), (0, 0, 2), True, True, {"b"}),
        "d7": ((None, 0, 0), (0, 0, 0), False, False, set()),
        "d8": ((None, 1, 1), (0, 1, 1), True, True, {"b′"}),
        "d9": ((None, 2, 2), (0, 2, 2), True, True, {"b′"}),
    }
    for label, (cand, end, conf, strong, ev) in exp.items():
        rec = run.rec(label)
        assert (rec.dc_cand, rec.rc_cand, rec.xc_cand) == cand, label
        assert (rec.dc, rec.rc, rec.xc) == end, label
        assert (rec.confirmed, rec.strong, rec.evidence) == (conf, strong, frozenset(ev)), label
    assert {e.key for e in run.events("d4")} == {
        (E.STABILIZED, eid, 1),
        (E.RECOVERY_OBS, eid, 1, 1),
    }
    assert run.types("d6") == [E.DOWNGRADE] and run.event("d6", E.DOWNGRADE).basis["reasons"] == [
        "b"
    ]
    assert_day(run, "d6", h_d=89.0, level=Level.WATCH, downgraded=True)
    # d7：关注，不解除；撤销（BELOW_SMA；周期 1，第 1 次）
    assert_day(run, "d7", h_d=89.0, giveback=0.5, level=Level.WATCH)
    revoked = run.event("d7", E.RECOVERY_REVOKED)
    assert revoked.identity == (eid, 1, 1) and revoked.basis["reason"] == "BELOW_SMA"
    assert run.types("d7") == [E.RECOVERY_REVOKED]
    assert {e.key for e in run.events("d8")} == {(E.RECOVERY_OBS, eid, 1, 2)}
    assert_day(run, "d8", h_d=91.0)
    assert_day(run, "d9", h_d=92.0)
    assert run.events("d9") == []  # 不得用"候选 + 1"提前解除
    rec10 = run.rec("d10")
    assert (rec10.rc_cand, rec10.xc_cand) == (3, 3) and rec10.display is Display.NORMAL
    rel = run.event("d10", E.RELEASE)
    assert rel.basis["reasons"] == ["DIRECT_STRONG_RECOVERY", "b′"]
    assert rel.basis["today_evidence"] == ["b′"] and rel.basis["strong"] is True
    assert run.types("d10") == [E.RELEASE]
    # B-01 推导（日末计数取自上表；H_d、确认状态与事件按规则）：d6 降级 H_d = 89；d7 撤销；
    # d8 同周期第 2 次确认，H_d = 91；d9 H_d = 92；d10 解除
    check_days(
        run,
        {
            "d1": started("d1", "A"),
            "d4": {
                "prev_confirm": True,
                "obs_issued": True,
                "dc": 1,
                "events": [("STAB", 1), ("OBS", 1, 1)],
            },
            "d5": {"dc": 2, "xc": 1},
            "d6": {
                "level": "W",
                "downgraded": True,
                "h_d": 89.0,
                "dc": 0,
                "rc": 0,
                "xc": 2,
                "events": [("DOWN",)],
            },
            "d7": {"prev_confirm": False, "xc": 0, "events": [("REV", 1, 1)]},
            "d8": {"prev_confirm": True, "h_d": 91.0, "rc": 1, "xc": 1, "events": [("OBS", 1, 2)]},
            "d9": {"h_d": 92.0, "rc": 2, "xc": 2},
            "d10": released("d10"),
        },
    )
    assert run.invariants() == []


def test_S17_consecutive_missing_after_confirmation() -> None:
    rows = [
        ("x0", 80, None),
        ("x1", 81, 83),
        ("x2", 82, 82.5),
        ("x3", 83, 82),
        ("x4", None, None),
        ("x5", None, None),
        ("x6", 84, 83),
        ("x7", 85, 83.2),
        ("x8", 86, 83.5),
    ]
    run = s_run(rows, params())
    eid = f"{SYMBOL}:{run.labels['x0']}"
    lab = run.labels
    exp = {  # m, cycle, obs_start, since, 确, dc, prev_confirm（日末）
        "x0": (0, 1, "x0", 0, None, 0, False),  # 启动日不判断恢复确认（记录为未计算）
        "x1": (0, 1, "x0", 1, False, 0, False),
        "x2": (0, 1, "x0", 2, False, 0, False),
        "x3": (0, 1, "x0", 3, True, 1, True),
        "x4": (1, 2, "x4", 0, None, 0, False),
        "x5": (2, 3, "x5", 0, None, 0, False),
        "x6": (0, 3, "x5", 1, False, 0, False),
        "x7": (0, 3, "x5", 2, False, 0, False),
        "x8": (0, 3, "x5", 3, True, 1, True),
    }
    for label, (m, cycle, obs, since, conf, dc, prev) in exp.items():
        assert_day(
            run,
            label,
            m=m,
            cycle_no=cycle,
            obs_start=lab[obs],
            since_obs=since,
            dc=dc,
            prev_confirm=prev,
            display=Display.ALERT,
            l_ep=80.0,
        )
        if conf is not None:
            assert run.rec(label).confirmed is conf, label
    assert {e.key for e in run.events("x3")} == {
        (E.STABILIZED, eid, 1),
        (E.RECOVERY_OBS, eid, 1, 1),
    }
    assert run.types("x4") == sorted([E.RECOVERY_REVOKED, E.DATA_ISSUE])
    gap = run.event("x4", E.RECOVERY_REVOKED)
    assert gap.identity == (eid, 1, 1) and gap.basis["reason"] == "DATA_GAP"  # 被终止的周期 1
    assert run.types("x5") == [E.DATA_ISSUE]  # 不再发撤销
    assert run.rec("x4").data_status is DataStatus.MISSING
    assert {e.key for e in run.events("x8")} == {
        (E.STABILIZED, eid, 3),
        (E.RECOVERY_OBS, eid, 3, 1),
    }
    # B-01 推导：x4 缺价撤销周期 1 的确认并重置（周期 2），x5 再重置（周期 3）；x8 周期 3 企稳并确认
    check_days(
        run,
        {
            "x0": started("x0", "A"),
            "x3": {
                "prev_confirm": True,
                "obs_issued": True,
                "dc": 1,
                "events": [("STAB", 1), ("OBS", 1, 1)],
            },
            "x4": {**reset_at("x4", 2), "events": [("REV", 1, 1), ("ISSUE",)]},
            "x5": {**reset_at("x5", 3), "events": [("ISSUE",)]},
            "x8": {
                "prev_confirm": True,
                "obs_issued": True,
                "dc": 1,
                "events": [("STAB", 3), ("OBS", 3, 1)],
            },
        },
    )
    assert run.invariants() == []


@pytest.mark.parametrize("release", [TWO, DIRECT])
def test_S18_no_progress_events_on_release_day(release: ReleasePath) -> None:
    rows = [("d1", 96, None), ("d2", 97, 99), ("d3", 97.5, 98.5), ("d4", 98, 97)]
    run = s_run(rows, params(release=release))
    assert run.rec("d1").level is Level.WATCH
    for label, since, conf, ev, rc in (("d2", 1, False, {"a′"}, 1), ("d3", 2, False, {"a′"}, 2)):
        assert_day(run, label, since_obs=since, confirmed=conf, evidence=frozenset(ev), rc=rc)
    assert_day(run, "d2", d_ep=1.5)
    rec = run.rec("d4")
    assert (rec.since_obs, rec.confirmed, rec.strong, rec.rc_cand) == (3, True, True, 3)
    assert rec.strong_threshold == pytest.approx(98.0)  # 96 + min(6, 2)
    assert rec.xc_cand == (1 if release is DIRECT else 0)
    assert run.types("d4") == [E.RELEASE]  # 只有 RELEASE
    assert run.event("d4", E.RELEASE).basis["reasons"] == ["a′", "b′"]
    # B-01 推导：d2、d3 证据 a′；d4 解除，当日确认成立但不发进度事件，
    # prev_confirm、obs_issued 保持为否
    check_days(
        run,
        {
            "d1": started("d1", "W"),
            "d2": {"rc": 1},
            "d3": {"rc": 2},
            "d4": released("d4"),
        },
    )
    assert run.invariants() == []


def s19_run(case: str) -> Run:
    """S19：ATR 每日给定为 2；历史长度充足；显式给出 r−1 及之前的窗口（各日开盘、收盘 = 100）。"""
    p = params()
    labels = [*(f"h{i}" for i in range(5)), "rm1", "r0", "r1"]
    days = weekdays(START_DAY, len(labels))
    lab = dict(zip(labels, days, strict=True))
    table = {lab[x]: TableRow(100, 100, atr=2) for x in labels[:-2]}
    if case != "b":
        table[lab["r0"]] = TableRow(100, 100, atr=2)
    table[lab["r1"]] = TableRow(93 if case in "ab" else 94, 97, atr=2)
    src = TableIndicators(
        TradingCalendar.from_days(days), table, n=p.candidate.n, prior_valid_bars=10_000
    )
    return Run(lab, run_with_source(SYMBOL, days, src, p), p)


@pytest.mark.parametrize("case", ["a", "b", "c"])
def test_S19_gap_start_and_missing_previous_day(case: str) -> None:
    run = s19_run(case)
    rec = run.rec("r1")
    if case == "b":
        assert run.types("r0") == [E.DATA_ISSUE] and run.rec("r0").data_status is DataStatus.MISSING
        assert (rec.r2_status, rec.r2_score) == ("MISSING_DATA", None)
        assert rec.d_t == pytest.approx(1.5) and rec.display is Display.NORMAL
        assert run.events("r1") == []
        # B-01 推导：r0 缺价（无预警，只有 DATA_ISSUE），r1 不启动
        check_days(run, {"r0": {"events": [("ISSUE",)]}})
        return
    assert run.rec("r0").display is Display.NORMAL and run.rec("r0").d_t == 0
    assert rec.r2_status == "OK" and rec.r2_score == pytest.approx(-3.5 if case == "a" else -3.0)
    start = run.event("r1", E.START)
    assert start.basis["reasons"] == ["GAP"] and start.transition == "T7"
    assert (rec.level, rec.p_ep, rec.p_ep_day, rec.atr_ep, rec.l_ep) == (
        Level.WATCH,
        100.0,
        run.labels["r0"],
        2.0,
        97.0,
    )
    assert rec.d_ep == pytest.approx(1.5)
    check_days(run, {"r1": started("r1", "W")})  # B-01 推导：r1 以关注启动，此前无预警
    assert run.invariants() == []


@pytest.mark.parametrize(("n", "short"), [(20, 82), (14, 76)])
def test_S21a_S21b_history_length(n: int, short: int) -> None:
    p = params(n=n)
    assert short + 1 == FIXED.w + n
    bars = f_bars([])[: short + 1]
    cal = TradingCalendar.from_days(b.day for b in bars)
    run = run_with_source(SYMBOL, cal.days, ComputedIndicators(cal, bars), p)
    recs = [r.record for r in run.results]
    assert recs[short - 1].display is Display.INSUFFICIENT  # 第 short 条：不可计算（S21a）
    assert recs[short].display is Display.NORMAL  # 第 short + 1 条：可计算（S21b）
    check_days(Run({}, run, p), {})  # B-01 推导：平盘，各日无预警、无事件


@pytest.mark.parametrize(("close", "starts"), [(96 + 1e-10, True), (96 + 4e-10, False)])
def test_S21c_S21d_start_boundary_eps_ratio(close: float, starts: bool) -> None:
    run = s_run([("q1", close, None)], params())
    assert (run.types("q1") == [E.START]) is starts
    check_days(run, {"q1": started("q1", "W")} if starts else {})  # B-01 推导：D_T ≈ 2，关注


@pytest.mark.parametrize(("close", "new_low"), [(80 - 5e-10, False), (80 - 2e-9, True)])
def test_S21e_S21f_new_low_boundary_eps_price(close: float, new_low: bool) -> None:
    run = s_run([("q0", 80, None), ("q1", close, None)], params())
    assert run.rec("q1").new_low is new_low
    assert run.rec("q1").cycle_no == (2 if new_low else 1)
    # B-01 推导：q0 以警戒启动（D_T = 10）；q1 为新低时观察重置进入周期 2
    check_days(run, {"q0": started("q0", "A"), **({"q1": reset_at("q1", 2)} if new_low else {})})


def test_S21h_quantized_input_starts(tmp_path: Path) -> None:
    """读取入口 + 引擎：当日复权收盘文本 "96.0000004" 量化为 96.000000，D_T = 2，START。"""
    from stock_risk.application.market_data_loader import MarketDataLoader
    from stock_risk.domain.lifecycle.runner import run_bars
    from stock_risk.infrastructure.storage.csv_raw_reader import CsvRawDataReader
    from stock_risk.infrastructure.storage.csv_raw_repository import BAR_COLUMNS

    class Fixed:
        def __init__(self, value) -> None:
            self.value = value

        def load(self):
            return self.value

    bars = f_bars([])  # 90 根平盘（峰值 100、ATR 2）
    day = weekdays(START_DAY, 91)[-1]
    raw = tmp_path / "raw"
    raw.mkdir()
    lines = [",".join(BAR_COLUMNS)]
    for b in bars:
        lines.append(f"{b.day},100,101,99,100,1,100,101,99,100")
    lines.append(f"{day},100,101,95,96.0000004,1,100,101,95,96.0000004")
    for sym in ("SPY", "X"):
        (raw / f"{sym}.csv").write_text("\n".join(lines) + "\n", "utf-8")
    data = MarketDataLoader(CsvRawDataReader(raw), Fixed({}), Fixed({})).load("X")
    assert data.history.bars[-1].adj_close == quantize_price("96.0000004") == 96.0
    p = params()
    cal = TradingCalendar.from_days(p_.day for p_ in data.presence)
    out = run_bars(SYMBOL, cal, data.presence, data.history.bars, p)
    last = out.results[-1]
    assert last.record.d_t == 2.0 and [e.type for e in last.events] == [E.START]
    # B-01 推导：此前各日无预警；最后一日以关注启动（D_T = 2）
    check_days(Run({"q": day}, out, p), {"q": started("q", "W")})
    # 对照：不量化时 D_T = 1.9999998，不启动
    assert 2 - p.tol.eps_ratio > (100 - 96.0000004) / 2


def test_S22_revoked_then_recovery_fail() -> None:
    rows = [
        ("y1", 80, None),
        ("y2", 81, 83),
        ("y3", 82, 82.5),
        ("y4", 83, 82),
        ("y5", 82, 82.5),
        ("y6", 79, 82),
    ]
    run = s_run(rows, params())
    eid = f"{SYMBOL}:{run.labels['y1']}"
    for label, l_ep, cycle, since, conf, issued, dc in (
        ("y2", 80, 1, 1, False, False, 0),
        ("y3", 80, 1, 2, False, False, 0),
        ("y4", 80, 1, 3, True, True, 1),
        ("y5", 80, 1, 4, False, True, 0),
        ("y6", 79, 2, 0, False, False, 0),
    ):
        assert_day(
            run,
            label,
            l_ep=float(l_ep),
            cycle_no=cycle,
            since_obs=since,
            confirmed=conf,
            obs_issued=issued,
            dc_cand=dc,
        )
    assert {e.key for e in run.events("y4")} == {
        (E.STABILIZED, eid, 1),
        (E.RECOVERY_OBS, eid, 1, 1),
    }
    revoked = run.event("y5", E.RECOVERY_REVOKED)
    assert revoked.identity == (eid, 1, 1) and revoked.basis["reason"] == "BELOW_SMA"
    fail = run.event("y6", E.RECOVERY_FAIL)
    assert fail.identity == (eid, 1) and fail.basis["reason"] == "NEW_LOW"
    assert run.types("y6") == [E.RECOVERY_FAIL]  # 不再发撤销
    # B-01 推导：y4 确认，y5 撤销；y6 新低进入周期 2，发出周期 1 的 RECOVERY_FAIL
    check_days(
        run,
        {
            "y1": started("y1", "A"),
            "y4": {
                "prev_confirm": True,
                "obs_issued": True,
                "dc": 1,
                "events": [("STAB", 1), ("OBS", 1, 1)],
            },
            "y5": {"prev_confirm": False, "dc": 0, "events": [("REV", 1, 1)]},
            "y6": {**reset_at("y6", 2), "events": [("FAIL", 1)]},
        },
    )
    assert run.invariants() == []


def test_I21_release_reason_is_todays_evidence_with_alternating_segment() -> None:
    """I21：rc 计数段内证据交替 {a′}、{a′}、{b′}，解除原因只记 b′，计数段证据为三天各自的集合。

    构造（推导）：沿用 S1 至 s6 降级（关注，L_ep = 78，H_d = 85）；
    r1、r2 收盘 97 而 SMA10 = 99（确认不成立，只有 a′：D_ep = 1.5 < 2）；
    r3 收盘 95、SMA10 = 90（确认与强恢复成立，D_ep = 2.5 不小于 2，只有 b′；
    回吐 (97 − 95) / 2 = 1 < 2）。
    """
    rows = [r for r in S1_ROWS if r[0] in {"s0", "s1", "s2", "s3", "s4", "s5", "s6"}]
    rows += [("r1", 97, 99), ("r2", 97, 99), ("r3", 95, 90)]
    run = s_run(rows, params())
    assert run.rec("s6").level is Level.WATCH
    assert [run.rec(x).evidence for x in ("r1", "r2", "r3")] == [
        frozenset({"a′"}),
        frozenset({"a′"}),
        frozenset({"b′"}),
    ]
    assert run.rec("r3").rc_cand == 3
    rel = run.event("r3", E.RELEASE)
    assert rel.basis["reasons"] == ["b′"]
    assert rel.basis["segment"] == [["a′"], ["a′"], ["b′"]]
    # B-01 推导：s0–s6 同 S1；r1 确认消失（撤销周期 2 第 1 次），H_d 升至 97；r3 解除
    s1_part = {k: v for k, v in S1_DAYS.items() if k in {"s0", "s1", "s4", "s5", "s6"}}
    check_days(
        run,
        {
            **s1_part,
            "r1": {"prev_confirm": False, "h_d": 97.0, "rc": 1, "events": [("REV", 2, 1)]},
            "r2": {"rc": 2},
            "r3": released("r3"),
        },
    )
    assert run.invariants() == []
